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
def _spider_idle(self) -> None:
    """
        Called when a spider gets idle, i.e. when there are no remaining requests to download or schedule.
        It can be called multiple times. If a handler for the spider_idle signal raises a DontCloseSpider
        exception, the spider is not closed until the next loop and this function is guaranteed to be called
        (at least) once again. A handler can raise CloseSpider to provide a custom closing reason.
        """
    assert self.spider is not None
    expected_ex = (DontCloseSpider, CloseSpider)
    res = self.signals.send_catch_log(signals.spider_idle, spider=self.spider, dont_log=expected_ex)
    detected_ex = {ex: x.value for _, x in res for ex in expected_ex if isinstance(x, Failure) and isinstance(x.value, ex)}
    if DontCloseSpider in detected_ex:
        return None
    if self.spider_is_idle():
        ex = detected_ex.get(CloseSpider, CloseSpider(reason='finished'))
        assert isinstance(ex, CloseSpider)
        self.close_spider(self.spider, reason=ex.reason)