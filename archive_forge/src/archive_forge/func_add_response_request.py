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
def add_response_request(self, result: Union[Response, Failure], request: Request) -> Deferred:
    deferred: Deferred = Deferred()
    self.queue.append((result, request, deferred))
    if isinstance(result, Response):
        self.active_size += max(len(result.body), self.MIN_RESPONSE_SIZE)
    else:
        self.active_size += self.MIN_RESPONSE_SIZE
    return deferred