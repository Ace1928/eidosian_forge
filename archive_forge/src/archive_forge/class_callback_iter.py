import warnings
from collections import Counter, defaultdict, deque, abc
from collections.abc import Sequence
from functools import partial, reduce, wraps
from heapq import heapify, heapreplace, heappop
from itertools import (
from math import exp, factorial, floor, log
from queue import Empty, Queue
from random import random, randrange, uniform
from operator import itemgetter, mul, sub, gt, lt, ge, le
from sys import hexversion, maxsize
from time import monotonic
from .recipes import (
class callback_iter:
    """Convert a function that uses callbacks to an iterator.

    Let *func* be a function that takes a `callback` keyword argument.
    For example:

    >>> def func(callback=None):
    ...     for i, c in [(1, 'a'), (2, 'b'), (3, 'c')]:
    ...         if callback:
    ...             callback(i, c)
    ...     return 4


    Use ``with callback_iter(func)`` to get an iterator over the parameters
    that are delivered to the callback.

    >>> with callback_iter(func) as it:
    ...     for args, kwargs in it:
    ...         print(args)
    (1, 'a')
    (2, 'b')
    (3, 'c')

    The function will be called in a background thread. The ``done`` property
    indicates whether it has completed execution.

    >>> it.done
    True

    If it completes successfully, its return value will be available
    in the ``result`` property.

    >>> it.result
    4

    Notes:

    * If the function uses some keyword argument besides ``callback``, supply
      *callback_kwd*.
    * If it finished executing, but raised an exception, accessing the
      ``result`` property will raise the same exception.
    * If it hasn't finished executing, accessing the ``result``
      property from within the ``with`` block will raise ``RuntimeError``.
    * If it hasn't finished executing, accessing the ``result`` property from
      outside the ``with`` block will raise a
      ``more_itertools.AbortThread`` exception.
    * Provide *wait_seconds* to adjust how frequently the it is polled for
      output.

    """

    def __init__(self, func, callback_kwd='callback', wait_seconds=0.1):
        self._func = func
        self._callback_kwd = callback_kwd
        self._aborted = False
        self._future = None
        self._wait_seconds = wait_seconds
        self._executor = __import__().futures.__import__('concurrent.futures').futures.ThreadPoolExecutor(max_workers=1)
        self._iterator = self._reader()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._aborted = True
        self._executor.shutdown()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    @property
    def done(self):
        if self._future is None:
            return False
        return self._future.done()

    @property
    def result(self):
        if not self.done:
            raise RuntimeError('Function has not yet completed')
        return self._future.result()

    def _reader(self):
        q = Queue()

        def callback(*args, **kwargs):
            if self._aborted:
                raise AbortThread('canceled by user')
            q.put((args, kwargs))
        self._future = self._executor.submit(self._func, **{self._callback_kwd: callback})
        while True:
            try:
                item = q.get(timeout=self._wait_seconds)
            except Empty:
                pass
            else:
                q.task_done()
                yield item
            if self._future.done():
                break
        remaining = []
        while True:
            try:
                item = q.get_nowait()
            except Empty:
                break
            else:
                q.task_done()
                remaining.append(item)
        q.join()
        yield from remaining