import base64
import functools
import hashlib
import logging
import mimetypes
import os
import time
from collections import defaultdict
from contextlib import suppress
from ftplib import FTP
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import DefaultDict, Optional, Set, Union
from urllib.parse import urlparse
from itemadapter import ItemAdapter
from twisted.internet import defer, threads
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.pipelines.media import MediaPipeline
from scrapy.settings import Settings
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.datatypes import CaseInsensitiveDict
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import md5sum
from scrapy.utils.python import to_bytes
from scrapy.utils.request import referer_str
class FilesPipeline(MediaPipeline):
    """Abstract pipeline that implement the file downloading

    This pipeline tries to minimize network transfers and file processing,
    doing stat of the files and determining if file is new, up-to-date or
    expired.

    ``new`` files are those that pipeline never processed and needs to be
        downloaded from supplier site the first time.

    ``uptodate`` files are the ones that the pipeline processed and are still
        valid files.

    ``expired`` files are those that pipeline already processed but the last
        modification was made long time ago, so a reprocessing is recommended to
        refresh it in case of change.

    """
    MEDIA_NAME = 'file'
    EXPIRES = 90
    STORE_SCHEMES = {'': FSFilesStore, 'file': FSFilesStore, 's3': S3FilesStore, 'gs': GCSFilesStore, 'ftp': FTPFilesStore}
    DEFAULT_FILES_URLS_FIELD = 'file_urls'
    DEFAULT_FILES_RESULT_FIELD = 'files'

    def __init__(self, store_uri, download_func=None, settings=None):
        store_uri = _to_string(store_uri)
        if not store_uri:
            raise NotConfigured
        if isinstance(settings, dict) or settings is None:
            settings = Settings(settings)
        cls_name = 'FilesPipeline'
        self.store = self._get_store(store_uri)
        resolve = functools.partial(self._key_for_pipe, base_class_name=cls_name, settings=settings)
        self.expires = settings.getint(resolve('FILES_EXPIRES'), self.EXPIRES)
        if not hasattr(self, 'FILES_URLS_FIELD'):
            self.FILES_URLS_FIELD = self.DEFAULT_FILES_URLS_FIELD
        if not hasattr(self, 'FILES_RESULT_FIELD'):
            self.FILES_RESULT_FIELD = self.DEFAULT_FILES_RESULT_FIELD
        self.files_urls_field = settings.get(resolve('FILES_URLS_FIELD'), self.FILES_URLS_FIELD)
        self.files_result_field = settings.get(resolve('FILES_RESULT_FIELD'), self.FILES_RESULT_FIELD)
        super().__init__(download_func=download_func, settings=settings)

    @classmethod
    def from_settings(cls, settings):
        s3store = cls.STORE_SCHEMES['s3']
        s3store.AWS_ACCESS_KEY_ID = settings['AWS_ACCESS_KEY_ID']
        s3store.AWS_SECRET_ACCESS_KEY = settings['AWS_SECRET_ACCESS_KEY']
        s3store.AWS_SESSION_TOKEN = settings['AWS_SESSION_TOKEN']
        s3store.AWS_ENDPOINT_URL = settings['AWS_ENDPOINT_URL']
        s3store.AWS_REGION_NAME = settings['AWS_REGION_NAME']
        s3store.AWS_USE_SSL = settings['AWS_USE_SSL']
        s3store.AWS_VERIFY = settings['AWS_VERIFY']
        s3store.POLICY = settings['FILES_STORE_S3_ACL']
        gcs_store = cls.STORE_SCHEMES['gs']
        gcs_store.GCS_PROJECT_ID = settings['GCS_PROJECT_ID']
        gcs_store.POLICY = settings['FILES_STORE_GCS_ACL'] or None
        ftp_store = cls.STORE_SCHEMES['ftp']
        ftp_store.FTP_USERNAME = settings['FTP_USER']
        ftp_store.FTP_PASSWORD = settings['FTP_PASSWORD']
        ftp_store.USE_ACTIVE_MODE = settings.getbool('FEED_STORAGE_FTP_ACTIVE')
        store_uri = settings['FILES_STORE']
        return cls(store_uri, settings=settings)

    def _get_store(self, uri: str):
        if Path(uri).is_absolute():
            scheme = 'file'
        else:
            scheme = urlparse(uri).scheme
        store_cls = self.STORE_SCHEMES[scheme]
        return store_cls(uri)

    def media_to_download(self, request, info, *, item=None):

        def _onsuccess(result):
            if not result:
                return
            last_modified = result.get('last_modified', None)
            if not last_modified:
                return
            age_seconds = time.time() - last_modified
            age_days = age_seconds / 60 / 60 / 24
            if age_days > self.expires:
                return
            referer = referer_str(request)
            logger.debug('File (uptodate): Downloaded %(medianame)s from %(request)s referred in <%(referer)s>', {'medianame': self.MEDIA_NAME, 'request': request, 'referer': referer}, extra={'spider': info.spider})
            self.inc_stats(info.spider, 'uptodate')
            checksum = result.get('checksum', None)
            return {'url': request.url, 'path': path, 'checksum': checksum, 'status': 'uptodate'}
        path = self.file_path(request, info=info, item=item)
        dfd = defer.maybeDeferred(self.store.stat_file, path, info)
        dfd.addCallbacks(_onsuccess, lambda _: None)
        dfd.addErrback(lambda f: logger.error(self.__class__.__name__ + '.store.stat_file', exc_info=failure_to_exc_info(f), extra={'spider': info.spider}))
        return dfd

    def media_failed(self, failure, request, info):
        if not isinstance(failure.value, IgnoreRequest):
            referer = referer_str(request)
            logger.warning('File (unknown-error): Error downloading %(medianame)s from %(request)s referred in <%(referer)s>: %(exception)s', {'medianame': self.MEDIA_NAME, 'request': request, 'referer': referer, 'exception': failure.value}, extra={'spider': info.spider})
        raise FileException

    def media_downloaded(self, response, request, info, *, item=None):
        referer = referer_str(request)
        if response.status != 200:
            logger.warning('File (code: %(status)s): Error downloading file from %(request)s referred in <%(referer)s>', {'status': response.status, 'request': request, 'referer': referer}, extra={'spider': info.spider})
            raise FileException('download-error')
        if not response.body:
            logger.warning('File (empty-content): Empty file from %(request)s referred in <%(referer)s>: no-content', {'request': request, 'referer': referer}, extra={'spider': info.spider})
            raise FileException('empty-content')
        status = 'cached' if 'cached' in response.flags else 'downloaded'
        logger.debug('File (%(status)s): Downloaded file from %(request)s referred in <%(referer)s>', {'status': status, 'request': request, 'referer': referer}, extra={'spider': info.spider})
        self.inc_stats(info.spider, status)
        try:
            path = self.file_path(request, response=response, info=info, item=item)
            checksum = self.file_downloaded(response, request, info, item=item)
        except FileException as exc:
            logger.warning('File (error): Error processing file from %(request)s referred in <%(referer)s>: %(errormsg)s', {'request': request, 'referer': referer, 'errormsg': str(exc)}, extra={'spider': info.spider}, exc_info=True)
            raise
        except Exception as exc:
            logger.error('File (unknown-error): Error processing file from %(request)s referred in <%(referer)s>', {'request': request, 'referer': referer}, exc_info=True, extra={'spider': info.spider})
            raise FileException(str(exc))
        return {'url': request.url, 'path': path, 'checksum': checksum, 'status': status}

    def inc_stats(self, spider, status):
        spider.crawler.stats.inc_value('file_count', spider=spider)
        spider.crawler.stats.inc_value(f'file_status_count/{status}', spider=spider)

    def get_media_requests(self, item, info):
        urls = ItemAdapter(item).get(self.files_urls_field, [])
        return [Request(u, callback=NO_CALLBACK) for u in urls]

    def file_downloaded(self, response, request, info, *, item=None):
        path = self.file_path(request, response=response, info=info, item=item)
        buf = BytesIO(response.body)
        checksum = md5sum(buf)
        buf.seek(0)
        self.store.persist_file(path, buf, info)
        return checksum

    def item_completed(self, results, item, info):
        with suppress(KeyError):
            ItemAdapter(item)[self.files_result_field] = [x for ok, x in results if ok]
        return item

    def file_path(self, request, response=None, info=None, *, item=None):
        media_guid = hashlib.sha1(to_bytes(request.url)).hexdigest()
        media_ext = Path(request.url).suffix
        if media_ext not in mimetypes.types_map:
            media_ext = ''
            media_type = mimetypes.guess_type(request.url)[0]
            if media_type:
                media_ext = mimetypes.guess_extension(media_type)
        return f'full/{media_guid}{media_ext}'