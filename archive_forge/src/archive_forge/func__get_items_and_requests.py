import inspect
import json
import logging
from typing import Dict
from itemadapter import ItemAdapter, is_item
from twisted.internet.defer import maybeDeferred
from w3lib.url import is_url
from scrapy.commands import BaseRunSpiderCommand
from scrapy.exceptions import UsageError
from scrapy.http import Request
from scrapy.utils import display
from scrapy.utils.asyncgen import collect_asyncgen
from scrapy.utils.defer import aiter_errback, deferred_from_coro
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import arg_to_iter
from scrapy.utils.spider import spidercls_for_request
def _get_items_and_requests(self, spider_output, opts, depth, spider, callback):
    items, requests = ([], [])
    for x in spider_output:
        if is_item(x):
            items.append(x)
        elif isinstance(x, Request):
            requests.append(x)
    return (items, requests, opts, depth, spider, callback)