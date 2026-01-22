import logging
from time import time
from typing import (
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.task import LoopingCall
from twisted.python.failure import Failure
from scrapy import signals
from scrapy.core.downloader import Downloader
from scrapy.core.scraper import Scraper
from scrapy.exceptions import CloseSpider, DontCloseSpider
from scrapy.http import Request, Response
from scrapy.logformatter import LogFormatter
from scrapy.settings import BaseSettings, Settings
from scrapy.signalmanager import SignalManager
from scrapy.spiders import Spider
from scrapy.utils.log import failure_to_exc_info, logformatter_adapter
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.reactor import CallLaterOnce
def _next_request_from_scheduler(self) -> Optional[Deferred]:
    assert self.slot is not None
    assert self.spider is not None
    request = self.slot.scheduler.next_request()
    if request is None:
        return None
    d = self._download(request)
    d.addBoth(self._handle_downloader_output, request)
    d.addErrback(lambda f: logger.info('Error while handling downloader output', exc_info=failure_to_exc_info(f), extra={'spider': self.spider}))

    def _remove_request(_: Any) -> None:
        assert self.slot
        self.slot.remove_request(request)
    d.addBoth(_remove_request)
    d.addErrback(lambda f: logger.info('Error while removing request from slot', exc_info=failure_to_exc_info(f), extra={'spider': self.spider}))
    slot = self.slot
    d.addBoth(lambda _: slot.nextcall.schedule())
    d.addErrback(lambda f: logger.info('Error while scheduling new request', exc_info=failure_to_exc_info(f), extra={'spider': self.spider}))
    return d