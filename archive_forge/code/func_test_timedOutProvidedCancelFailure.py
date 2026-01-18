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
def test_timedOutProvidedCancelFailure(self) -> None:
    """
        If a cancellation function is provided when the L{Deferred} is
        initialized, the L{Deferred} returns the cancellation value's
        non-L{CanceledError} failure when the L{Deferred} times out.
        """
    clock = Clock()
    error = ValueError('what!')
    d: Deferred[None] = Deferred(lambda c: c.errback(error))
    d.addTimeout(10, clock)
    self.assertNoResult(d)
    clock.advance(15)
    f = self.failureResultOf(d, ValueError)
    self.assertIs(f.value, error)