import functools
import logging
from collections import defaultdict
from inspect import signature
from warnings import warn
from twisted.internet.defer import Deferred, DeferredList
from twisted.python.failure import Failure
from scrapy.http.request import NO_CALLBACK
from scrapy.settings import Settings
from scrapy.utils.datatypes import SequenceExclude
from scrapy.utils.defer import defer_result, mustbe_deferred
from scrapy.utils.deprecate import ScrapyDeprecationWarning
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import arg_to_iter
def _modify_media_request(self, request):
    if self.handle_httpstatus_list:
        request.meta['handle_httpstatus_list'] = self.handle_httpstatus_list
    else:
        request.meta['handle_httpstatus_all'] = True