from __future__ import annotations
import re
from typing import Callable
from twisted.conch.insults import helper
from twisted.conch.insults.insults import (
from twisted.python import failure
from twisted.trial import unittest
class FakeScheduler:

    def __init__(self) -> None:
        self.calls: list[FakeDelayedCall] = []

    def callLater(self, timeout: float, f: Callable[..., None], *a: object, **kw: object) -> FakeDelayedCall:
        self.calls.append(FakeDelayedCall(self, timeout, f, a, kw))
        return self.calls[-1]