import collections
import datetime
import heapq
from tornado import gen, ioloop
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.locks import Event
from typing import Union, TypeVar, Generic, Awaitable, Optional
import typing
def _consume_expired(self) -> None:
    while self._putters and self._putters[0][1].done():
        self._putters.popleft()
    while self._getters and self._getters[0].done():
        self._getters.popleft()