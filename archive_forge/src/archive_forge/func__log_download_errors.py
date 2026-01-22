from __future__ import annotations
import logging
from collections import deque
from typing import (
from itemadapter import is_item
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Spider, signals
from scrapy.core.spidermw import SpiderMiddlewareManager
from scrapy.exceptions import CloseSpider, DropItem, IgnoreRequest
from scrapy.http import Request, Response
from scrapy.logformatter import LogFormatter
from scrapy.pipelines import ItemPipelineManager
from scrapy.signalmanager import SignalManager
from scrapy.utils.defer import (
from scrapy.utils.log import failure_to_exc_info, logformatter_adapter
from scrapy.utils.misc import load_object, warn_on_generator_with_return_value
from scrapy.utils.spider import iterate_spider_output
def _log_download_errors(self, spider_failure: Failure, download_failure: Failure, request: Request, spider: Spider) -> Union[Failure, None]:
    """Log and silence errors that come from the engine (typically download
        errors that got propagated thru here).

        spider_failure: the value passed into the errback of self.call_spider()
        download_failure: the value passed into _scrape2() from
        ExecutionEngine._handle_downloader_output() as "result"
        """
    if not download_failure.check(IgnoreRequest):
        if download_failure.frames:
            logkws = self.logformatter.download_error(download_failure, request, spider)
            logger.log(*logformatter_adapter(logkws), extra={'spider': spider}, exc_info=failure_to_exc_info(download_failure))
        else:
            errmsg = download_failure.getErrorMessage()
            if errmsg:
                logkws = self.logformatter.download_error(download_failure, request, spider, errmsg)
                logger.log(*logformatter_adapter(logkws), extra={'spider': spider})
    if spider_failure is not download_failure:
        return spider_failure
    return None