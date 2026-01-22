import collections.abc
import contextlib
import datetime
import functools
import numbers
import time
import jaraco.functools
class IntervalGovernor:
    """
    Decorate a function to only allow it to be called once per
    min_interval. Otherwise, it returns None.

    >>> gov = IntervalGovernor(30)
    >>> gov.min_interval.total_seconds()
    30.0
    """

    def __init__(self, min_interval):
        if isinstance(min_interval, numbers.Number):
            min_interval = datetime.timedelta(seconds=min_interval)
        self.min_interval = min_interval
        self.last_call = None

    def decorate(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            allow = not self.last_call or self.last_call.split() > self.min_interval
            if allow:
                self.last_call = Stopwatch()
                return func(*args, **kwargs)
        return wrapper
    __call__ = decorate