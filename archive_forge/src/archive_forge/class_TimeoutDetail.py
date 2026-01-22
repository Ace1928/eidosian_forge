from __future__ import annotations
import dataclasses
import datetime
import functools
import os
import signal
import time
import typing as t
from .io import (
from .config import (
from .util import (
from .thread import (
from .constants import (
from .test import (
@dataclasses.dataclass(frozen=True)
class TimeoutDetail:
    """Details required to enforce a timeout on test execution."""
    _DEADLINE_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
    deadline: datetime.datetime
    duration: int | float

    @property
    def remaining(self) -> datetime.timedelta:
        """The amount of time remaining before the timeout occurs. If the timeout has passed, this will be a negative duration."""
        return self.deadline - datetime.datetime.now(tz=datetime.timezone.utc).replace(microsecond=0)

    def to_dict(self) -> dict[str, t.Any]:
        """Return timeout details as a dictionary suitable for JSON serialization."""
        return dict(deadline=self.deadline.strftime(self._DEADLINE_FORMAT), duration=self.duration)

    @staticmethod
    def from_dict(value: dict[str, t.Any]) -> TimeoutDetail:
        """Return a TimeoutDetail instance using the value previously returned by to_dict."""
        return TimeoutDetail(deadline=datetime.datetime.strptime(value['deadline'], TimeoutDetail._DEADLINE_FORMAT).replace(tzinfo=datetime.timezone.utc), duration=value['duration'])

    @staticmethod
    def create(duration: int | float) -> TimeoutDetail | None:
        """Return a new TimeoutDetail instance for the specified duration (in minutes), or None if the duration is zero."""
        if not duration:
            return None
        if duration == int(duration):
            duration = int(duration)
        return TimeoutDetail(deadline=datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0) + datetime.timedelta(seconds=int(duration * 60)), duration=duration)