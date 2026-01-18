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
def test_pausedDeferredChained(self) -> None:
    """
        A paused Deferred encountered while pushing a result forward through a
        chain does not prevent earlier Deferreds from continuing to execute
        their callbacks.
        """
    first: Deferred[None] = Deferred()
    second: Deferred[None] = Deferred()
    first.addCallback(lambda ignored: second)
    first.callback(None)
    first.pause()
    second.callback(None)
    result: List[None] = []
    second.addCallback(result.append)
    self.assertEqual(result, [None])