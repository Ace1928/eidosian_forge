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
def test_synchronousImplicitErrorChain(self) -> None:
    """
        If a first L{Deferred} with a L{Failure} result is returned from a
        callback on a second L{Deferred}, the first L{Deferred}'s result is
        converted to L{None} and no unhandled error is logged when it is
        garbage collected.
        """
    first = defer.fail(RuntimeError("First Deferred's Failure"))

    def cb(_: None, first: Deferred[None]=first) -> Deferred[None]:
        return first
    second: Deferred[None] = Deferred()
    second.addCallback(cb)
    second.callback(None)
    firstResult: List[None] = []
    first.addCallback(firstResult.append)
    self.assertIsNone(firstResult[0])
    self.assertImmediateFailure(second, RuntimeError)