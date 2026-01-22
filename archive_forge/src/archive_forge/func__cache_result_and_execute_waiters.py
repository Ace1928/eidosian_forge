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
def _cache_result_and_execute_waiters(self, result, fp, info):
    if isinstance(result, Failure):
        result.cleanFailure()
        result.frames = []
        result.stack = None
        context = getattr(result.value, '__context__', None)
        if isinstance(context, StopIteration):
            setattr(result.value, '__context__', None)
    info.downloading.remove(fp)
    info.downloaded[fp] = result
    for wad in info.waiting.pop(fp):
        defer_result(result).chainDeferred(wad)