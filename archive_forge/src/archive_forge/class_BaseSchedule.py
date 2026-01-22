from __future__ import annotations
import re
from bisect import bisect, bisect_left
from collections import namedtuple
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo
from typing import Any, Callable, Mapping, Sequence
from kombu.utils.objects import cached_property
from celery import Celery
from . import current_app
from .utils.collections import AttributeDict
from .utils.time import (ffwd, humanize_seconds, localize, maybe_make_aware, maybe_timedelta, remaining, timezone,
class BaseSchedule:

    def __init__(self, nowfun: Callable | None=None, app: Celery | None=None):
        self.nowfun = nowfun
        self._app = app

    def now(self) -> datetime:
        return (self.nowfun or self.app.now)()

    def remaining_estimate(self, last_run_at: datetime) -> timedelta:
        raise NotImplementedError()

    def is_due(self, last_run_at: datetime) -> tuple[bool, datetime]:
        raise NotImplementedError()

    def maybe_make_aware(self, dt: datetime, naive_as_utc: bool=True) -> datetime:
        return maybe_make_aware(dt, self.tz, naive_as_utc=naive_as_utc)

    @property
    def app(self) -> Celery:
        return self._app or current_app._get_current_object()

    @app.setter
    def app(self, app: Celery) -> None:
        self._app = app

    @cached_property
    def tz(self) -> tzinfo:
        return self.app.timezone

    @cached_property
    def utc_enabled(self) -> bool:
        return self.app.conf.enable_utc

    def to_local(self, dt: datetime) -> datetime:
        if not self.utc_enabled:
            return timezone.to_local_fallback(dt)
        return dt

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BaseSchedule):
            return other.nowfun == self.nowfun
        return NotImplemented