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
def _process_spidermw_output(self, output: Any, request: Request, response: Response, spider: Spider) -> Optional[Deferred]:
    """Process each Request/Item (given in the output parameter) returned
        from the given spider
        """
    assert self.slot is not None
    if isinstance(output, Request):
        assert self.crawler.engine is not None
        self.crawler.engine.crawl(request=output)
    elif is_item(output):
        self.slot.itemproc_size += 1
        dfd = self.itemproc.process_item(output, spider)
        dfd.addBoth(self._itemproc_finished, output, response, spider)
        return dfd
    elif output is None:
        pass
    else:
        typename = type(output).__name__
        logger.error('Spider must return request, item, or None, got %(typename)r in %(request)s', {'request': request, 'typename': typename}, extra={'spider': spider})
    return None