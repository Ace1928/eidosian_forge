import functools
import sys
import threading
import time
import typing as t
import warnings
from abc import ABC, abstractmethod
from concurrent import futures
from inspect import iscoroutinefunction
from .retry import retry_base  # noqa
from .retry import retry_all  # noqa
from .retry import retry_always  # noqa
from .retry import retry_any  # noqa
from .retry import retry_if_exception  # noqa
from .retry import retry_if_exception_type  # noqa
from .retry import retry_if_exception_cause_type  # noqa
from .retry import retry_if_not_exception_type  # noqa
from .retry import retry_if_not_result  # noqa
from .retry import retry_if_result  # noqa
from .retry import retry_never  # noqa
from .retry import retry_unless_exception_type  # noqa
from .retry import retry_if_exception_message  # noqa
from .retry import retry_if_not_exception_message  # noqa
from .nap import sleep  # noqa
from .nap import sleep_using_event  # noqa
from .stop import stop_after_attempt  # noqa
from .stop import stop_after_delay  # noqa
from .stop import stop_all  # noqa
from .stop import stop_any  # noqa
from .stop import stop_never  # noqa
from .stop import stop_when_event_set  # noqa
from .wait import wait_chain  # noqa
from .wait import wait_combine  # noqa
from .wait import wait_exponential  # noqa
from .wait import wait_fixed  # noqa
from .wait import wait_incrementing  # noqa
from .wait import wait_none  # noqa
from .wait import wait_random  # noqa
from .wait import wait_random_exponential  # noqa
from .wait import wait_random_exponential as wait_full_jitter  # noqa
from .wait import wait_exponential_jitter  # noqa
from .before import before_log  # noqa
from .before import before_nothing  # noqa
from .after import after_log  # noqa
from .after import after_nothing  # noqa
from .before_sleep import before_sleep_log  # noqa
from .before_sleep import before_sleep_nothing  # noqa
from pip._vendor.tenacity._asyncio import AsyncRetrying  # noqa:E402,I100
class RetryCallState:
    """State related to a single call wrapped with Retrying."""

    def __init__(self, retry_object: BaseRetrying, fn: t.Optional[WrappedFn], args: t.Any, kwargs: t.Any) -> None:
        self.start_time = time.monotonic()
        self.retry_object = retry_object
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.attempt_number: int = 1
        self.outcome: t.Optional[Future] = None
        self.outcome_timestamp: t.Optional[float] = None
        self.idle_for: float = 0.0
        self.next_action: t.Optional[RetryAction] = None

    @property
    def seconds_since_start(self) -> t.Optional[float]:
        if self.outcome_timestamp is None:
            return None
        return self.outcome_timestamp - self.start_time

    def prepare_for_next_attempt(self) -> None:
        self.outcome = None
        self.outcome_timestamp = None
        self.attempt_number += 1
        self.next_action = None

    def set_result(self, val: t.Any) -> None:
        ts = time.monotonic()
        fut = Future(self.attempt_number)
        fut.set_result(val)
        self.outcome, self.outcome_timestamp = (fut, ts)

    def set_exception(self, exc_info: t.Tuple[t.Type[BaseException], BaseException, 'types.TracebackType| None']) -> None:
        ts = time.monotonic()
        fut = Future(self.attempt_number)
        fut.set_exception(exc_info[1])
        self.outcome, self.outcome_timestamp = (fut, ts)

    def __repr__(self) -> str:
        if self.outcome is None:
            result = 'none yet'
        elif self.outcome.failed:
            exception = self.outcome.exception()
            result = f'failed ({exception.__class__.__name__} {exception})'
        else:
            result = f'returned {self.outcome.result()}'
        slept = float(round(self.idle_for, 2))
        clsname = self.__class__.__name__
        return f'<{clsname} {id(self)}: attempt #{self.attempt_number}; slept for {slept}; last result: {result}>'