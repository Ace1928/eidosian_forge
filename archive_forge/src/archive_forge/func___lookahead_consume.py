import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def __lookahead_consume(self, limit=None):
    if not self.__done and (limit is None or limit > 0):
        it = iter(self.__it)
        try:
            now = next(it)
        except StopIteration:
            return
        self.__consumed.append(now)
        while not self.__done:
            try:
                next_ = next(it)
                self.__consumed.append(next_)
            except StopIteration:
                self.__done = True
                break
            finally:
                yield now
            now = next_
            if limit is not None:
                limit -= 1
                if limit <= 0:
                    break