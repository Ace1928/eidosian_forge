import logging
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse
from twisted.internet import defer, threads
from twisted.internet.defer import DeferredList
from w3lib.url import file_uri_to_path
from zope.interface import Interface, implementer
from scrapy import Spider, signals
from scrapy.exceptions import NotConfigured, ScrapyDeprecationWarning
from scrapy.extensions.postprocessing import PostProcessingManager
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.conf import feed_complete_default_values_from_settings
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.deprecate import create_deprecated_class
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.python import get_func_args, without_none_values
class FeedExporter:
    _pending_deferreds: List[defer.Deferred] = []

    @classmethod
    def from_crawler(cls, crawler):
        exporter = cls(crawler)
        crawler.signals.connect(exporter.open_spider, signals.spider_opened)
        crawler.signals.connect(exporter.close_spider, signals.spider_closed)
        crawler.signals.connect(exporter.item_scraped, signals.item_scraped)
        return exporter

    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.feeds = {}
        self.slots = []
        self.filters = {}
        if not self.settings['FEEDS'] and (not self.settings['FEED_URI']):
            raise NotConfigured
        if self.settings['FEED_URI']:
            warnings.warn('The `FEED_URI` and `FEED_FORMAT` settings have been deprecated in favor of the `FEEDS` setting. Please see the `FEEDS` setting docs for more details', category=ScrapyDeprecationWarning, stacklevel=2)
            uri = self.settings['FEED_URI']
            uri = str(uri) if not isinstance(uri, Path) else uri.absolute().as_uri()
            feed_options = {'format': self.settings.get('FEED_FORMAT', 'jsonlines')}
            self.feeds[uri] = feed_complete_default_values_from_settings(feed_options, self.settings)
            self.filters[uri] = self._load_filter(feed_options)
        for uri, feed_options in self.settings.getdict('FEEDS').items():
            uri = str(uri) if not isinstance(uri, Path) else uri.absolute().as_uri()
            self.feeds[uri] = feed_complete_default_values_from_settings(feed_options, self.settings)
            self.filters[uri] = self._load_filter(feed_options)
        self.storages = self._load_components('FEED_STORAGES')
        self.exporters = self._load_components('FEED_EXPORTERS')
        for uri, feed_options in self.feeds.items():
            if not self._storage_supported(uri, feed_options):
                raise NotConfigured
            if not self._settings_are_valid():
                raise NotConfigured
            if not self._exporter_supported(feed_options['format']):
                raise NotConfigured

    def open_spider(self, spider):
        for uri, feed_options in self.feeds.items():
            uri_params = self._get_uri_params(spider, feed_options['uri_params'])
            self.slots.append(self._start_new_batch(batch_id=1, uri=uri % uri_params, feed_options=feed_options, spider=spider, uri_template=uri))

    async def close_spider(self, spider):
        for slot in self.slots:
            self._close_slot(slot, spider)
        if self._pending_deferreds:
            await maybe_deferred_to_future(DeferredList(self._pending_deferreds))
        await maybe_deferred_to_future(self.crawler.signals.send_catch_log_deferred(signals.feed_exporter_closed))

    def _close_slot(self, slot, spider):

        def get_file(slot_):
            if isinstance(slot_.file, PostProcessingManager):
                slot_.file.close()
                return slot_.file.file
            return slot_.file
        if slot.itemcount:
            slot.finish_exporting()
        elif slot.store_empty and slot.batch_id == 1:
            slot.start_exporting()
            slot.finish_exporting()
        else:
            return None
        logmsg = f'{slot.format} feed ({slot.itemcount} items) in: {slot.uri}'
        d = defer.maybeDeferred(slot.storage.store, get_file(slot))
        d.addCallback(self._handle_store_success, logmsg, spider, type(slot.storage).__name__)
        d.addErrback(self._handle_store_error, logmsg, spider, type(slot.storage).__name__)
        self._pending_deferreds.append(d)
        d.addCallback(lambda _: self.crawler.signals.send_catch_log_deferred(signals.feed_slot_closed, slot=slot))
        d.addBoth(lambda _: self._pending_deferreds.remove(d))
        return d

    def _handle_store_error(self, f, logmsg, spider, slot_type):
        logger.error('Error storing %s', logmsg, exc_info=failure_to_exc_info(f), extra={'spider': spider})
        self.crawler.stats.inc_value(f'feedexport/failed_count/{slot_type}')

    def _handle_store_success(self, f, logmsg, spider, slot_type):
        logger.info('Stored %s', logmsg, extra={'spider': spider})
        self.crawler.stats.inc_value(f'feedexport/success_count/{slot_type}')

    def _start_new_batch(self, batch_id, uri, feed_options, spider, uri_template):
        """
        Redirect the output data stream to a new file.
        Execute multiple times if FEED_EXPORT_BATCH_ITEM_COUNT setting or FEEDS.batch_item_count is specified
        :param batch_id: sequence number of current batch
        :param uri: uri of the new batch to start
        :param feed_options: dict with parameters of feed
        :param spider: user spider
        :param uri_template: template of uri which contains %(batch_time)s or %(batch_id)d to create new uri
        """
        storage = self._get_storage(uri, feed_options)
        slot = FeedSlot(storage=storage, uri=uri, format=feed_options['format'], store_empty=feed_options['store_empty'], batch_id=batch_id, uri_template=uri_template, filter=self.filters[uri_template], feed_options=feed_options, spider=spider, exporters=self.exporters, settings=self.settings, crawler=getattr(self, 'crawler', None))
        return slot

    def item_scraped(self, item, spider):
        slots = []
        for slot in self.slots:
            if not slot.filter.accepts(item):
                slots.append(slot)
                continue
            slot.start_exporting()
            slot.exporter.export_item(item)
            slot.itemcount += 1
            if self.feeds[slot.uri_template]['batch_item_count'] and slot.itemcount >= self.feeds[slot.uri_template]['batch_item_count']:
                uri_params = self._get_uri_params(spider, self.feeds[slot.uri_template]['uri_params'], slot)
                self._close_slot(slot, spider)
                slots.append(self._start_new_batch(batch_id=slot.batch_id + 1, uri=slot.uri_template % uri_params, feed_options=self.feeds[slot.uri_template], spider=spider, uri_template=slot.uri_template))
            else:
                slots.append(slot)
        self.slots = slots

    def _load_components(self, setting_prefix):
        conf = without_none_values(self.settings.getwithbase(setting_prefix))
        d = {}
        for k, v in conf.items():
            try:
                d[k] = load_object(v)
            except NotConfigured:
                pass
        return d

    def _exporter_supported(self, format):
        if format in self.exporters:
            return True
        logger.error('Unknown feed format: %(format)s', {'format': format})

    def _settings_are_valid(self):
        """
        If FEED_EXPORT_BATCH_ITEM_COUNT setting or FEEDS.batch_item_count is specified uri has to contain
        %(batch_time)s or %(batch_id)d to distinguish different files of partial output
        """
        for uri_template, values in self.feeds.items():
            if values['batch_item_count'] and (not re.search('%\\(batch_time\\)s|%\\(batch_id\\)', uri_template)):
                logger.error('%%(batch_time)s or %%(batch_id)d must be in the feed URI (%s) if FEED_EXPORT_BATCH_ITEM_COUNT setting or FEEDS.batch_item_count is specified and greater than 0. For more info see: https://docs.scrapy.org/en/latest/topics/feed-exports.html#feed-export-batch-item-count', uri_template)
                return False
        return True

    def _storage_supported(self, uri, feed_options):
        scheme = urlparse(uri).scheme
        if scheme in self.storages or PureWindowsPath(uri).drive:
            try:
                self._get_storage(uri, feed_options)
                return True
            except NotConfigured as e:
                logger.error('Disabled feed storage scheme: %(scheme)s. Reason: %(reason)s', {'scheme': scheme, 'reason': str(e)})
        else:
            logger.error('Unknown feed storage scheme: %(scheme)s', {'scheme': scheme})

    def _get_storage(self, uri, feed_options):
        """Fork of create_instance specific to feed storage classes

        It supports not passing the *feed_options* parameters to classes that
        do not support it, and issuing a deprecation warning instead.
        """
        feedcls = self.storages.get(urlparse(uri).scheme, self.storages['file'])
        crawler = getattr(self, 'crawler', None)

        def build_instance(builder, *preargs):
            return build_storage(builder, uri, feed_options=feed_options, preargs=preargs)
        if crawler and hasattr(feedcls, 'from_crawler'):
            instance = build_instance(feedcls.from_crawler, crawler)
            method_name = 'from_crawler'
        elif hasattr(feedcls, 'from_settings'):
            instance = build_instance(feedcls.from_settings, self.settings)
            method_name = 'from_settings'
        else:
            instance = build_instance(feedcls)
            method_name = '__new__'
        if instance is None:
            raise TypeError(f'{feedcls.__qualname__}.{method_name} returned None')
        return instance

    def _get_uri_params(self, spider: Spider, uri_params_function: Optional[Union[str, Callable[[dict, Spider], dict]]], slot: Optional[FeedSlot]=None) -> dict:
        params = {}
        for k in dir(spider):
            params[k] = getattr(spider, k)
        utc_now = datetime.now(tz=timezone.utc)
        params['time'] = utc_now.replace(microsecond=0).isoformat().replace(':', '-')
        params['batch_time'] = utc_now.isoformat().replace(':', '-')
        params['batch_id'] = slot.batch_id + 1 if slot is not None else 1
        uripar_function = load_object(uri_params_function) if uri_params_function else lambda params, _: params
        new_params = uripar_function(params, spider)
        return new_params if new_params is not None else params

    def _load_filter(self, feed_options):
        item_filter_class = load_object(feed_options.get('item_filter', ItemFilter))
        return item_filter_class(feed_options)