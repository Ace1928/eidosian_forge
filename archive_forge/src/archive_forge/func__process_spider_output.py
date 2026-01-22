import logging
from inspect import isasyncgenfunction, iscoroutine
from itertools import islice
from typing import (
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.python.failure import Failure
from scrapy import Request, Spider
from scrapy.exceptions import _InvalidOutput
from scrapy.http import Response
from scrapy.middleware import MiddlewareManager
from scrapy.settings import BaseSettings
from scrapy.utils.asyncgen import as_async_generator, collect_asyncgen
from scrapy.utils.conf import build_component_list
from scrapy.utils.defer import (
from scrapy.utils.python import MutableAsyncChain, MutableChain
@inlineCallbacks
def _process_spider_output(self, response: Response, spider: Spider, result: Union[Iterable, AsyncIterable], start_index: int=0) -> Generator[Deferred, Any, Union[MutableChain, MutableAsyncChain]]:
    recovered: Union[MutableChain, MutableAsyncChain]
    last_result_is_async = isinstance(result, AsyncIterable)
    if last_result_is_async:
        recovered = MutableAsyncChain()
    else:
        recovered = MutableChain()
    method_list = islice(self.methods['process_spider_output'], start_index, None)
    for method_index, method_pair in enumerate(method_list, start=start_index):
        if method_pair is None:
            continue
        need_upgrade = need_downgrade = False
        if isinstance(method_pair, tuple):
            method_sync, method_async = method_pair
            method = method_async if last_result_is_async else method_sync
        else:
            method = method_pair
            if not last_result_is_async and isasyncgenfunction(method):
                need_upgrade = True
            elif last_result_is_async and (not isasyncgenfunction(method)):
                need_downgrade = True
        try:
            if need_upgrade:
                result = as_async_generator(result)
            elif need_downgrade:
                if not self.downgrade_warning_done:
                    logger.warning(f'Async iterable passed to {method.__qualname__} was downgraded to a non-async one')
                    self.downgrade_warning_done = True
                assert isinstance(result, AsyncIterable)
                result = (yield deferred_from_coro(collect_asyncgen(result)))
                if isinstance(recovered, AsyncIterable):
                    recovered_collected = (yield deferred_from_coro(collect_asyncgen(recovered)))
                    recovered = MutableChain(recovered_collected)
            result = method(response=response, result=result, spider=spider)
        except Exception as ex:
            exception_result = self._process_spider_exception(response, spider, Failure(ex), method_index + 1)
            if isinstance(exception_result, Failure):
                raise
            return exception_result
        if _isiterable(result):
            result = self._evaluate_iterable(response, spider, result, method_index + 1, recovered)
        else:
            if iscoroutine(result):
                result.close()
                msg = f'{method.__qualname__} must be an asynchronous generator (i.e. use yield)'
            else:
                msg = f'{method.__qualname__} must return an iterable, got {type(result)}'
            raise _InvalidOutput(msg)
        last_result_is_async = isinstance(result, AsyncIterable)
    if last_result_is_async:
        return MutableAsyncChain(result, recovered)
    return MutableChain(result, recovered)