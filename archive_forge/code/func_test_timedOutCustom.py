from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
def test_timedOutCustom(self) -> None:
    """
        If a custom C{onTimeoutCancel] function is provided, the
        L{Deferred} returns the custom function's return value if the
        L{Deferred} times out before callbacking or errbacking.
        The custom C{onTimeoutCancel} function can return a result instead of
        a failure.
        """
    clock = Clock()
    d: Deferred[None] = Deferred()
    d.addTimeout(10, clock, onTimeoutCancel=_overrideFunc)
    self.assertNoResult(d)
    clock.advance(15)
    self.assertEqual('OVERRIDDEN', self.successResultOf(d))