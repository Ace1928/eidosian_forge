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
def test_successResultBeforeTimeoutCustom(self) -> None:
    """
        The L{Deferred} callbacks with the result if it succeeds before
        the timeout, even if a custom C{onTimeoutCancel} function is provided.
        No cancellation happens after the callback either, which could also
        cancel inner deferreds.
        """
    clock = Clock()
    d: Deferred[str] = Deferred()
    d.addTimeout(10, clock, onTimeoutCancel=_overrideFunc)
    innerDeferred: Deferred[None] = Deferred()
    dCallbacked: Optional[str] = None

    def onCallback(results: str) -> Deferred[None]:
        nonlocal dCallbacked
        dCallbacked = results
        return innerDeferred
    d.addCallback(onCallback)
    d.callback('results')
    self.assertIsNot(None, dCallbacked)
    self.assertEqual(dCallbacked, 'results')
    clock.advance(15)
    self.assertNoResult(innerDeferred)