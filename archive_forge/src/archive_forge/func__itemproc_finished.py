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
def _itemproc_finished(self, output: Any, item: Any, response: Response, spider: Spider) -> Deferred:
    """ItemProcessor finished for the given ``item`` and returned ``output``"""
    assert self.slot is not None
    self.slot.itemproc_size -= 1
    if isinstance(output, Failure):
        ex = output.value
        if isinstance(ex, DropItem):
            logkws = self.logformatter.dropped(item, ex, response, spider)
            if logkws is not None:
                logger.log(*logformatter_adapter(logkws), extra={'spider': spider})
            return self.signals.send_catch_log_deferred(signal=signals.item_dropped, item=item, response=response, spider=spider, exception=output.value)
        assert ex
        logkws = self.logformatter.item_error(item, ex, response, spider)
        logger.log(*logformatter_adapter(logkws), extra={'spider': spider}, exc_info=failure_to_exc_info(output))
        return self.signals.send_catch_log_deferred(signal=signals.item_error, item=item, response=response, spider=spider, failure=output)
    logkws = self.logformatter.scraped(output, response, spider)
    if logkws is not None:
        logger.log(*logformatter_adapter(logkws), extra={'spider': spider})
    return self.signals.send_catch_log_deferred(signal=signals.item_scraped, item=output, response=response, spider=spider)