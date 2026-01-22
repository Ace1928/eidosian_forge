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
def _next_request(self) -> None:
    if self.slot is None:
        return
    assert self.spider is not None
    if self.paused:
        return None
    while not self._needs_backout() and self._next_request_from_scheduler() is not None:
        pass
    if self.slot.start_requests is not None and (not self._needs_backout()):
        try:
            request = next(self.slot.start_requests)
        except StopIteration:
            self.slot.start_requests = None
        except Exception:
            self.slot.start_requests = None
            logger.error('Error while obtaining start requests', exc_info=True, extra={'spider': self.spider})
        else:
            self.crawl(request)
    if self.spider_is_idle() and self.slot.close_if_idle:
        self._spider_idle()