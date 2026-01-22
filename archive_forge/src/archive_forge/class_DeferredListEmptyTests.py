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
class DeferredListEmptyTests(unittest.SynchronousTestCase):

    def setUp(self) -> None:
        self.callbackRan = 0

    def testDeferredListEmpty(self) -> None:
        """Testing empty DeferredList."""
        dl: Deferred[_DeferredListResultListT[object]] = DeferredList([])
        dl.addCallback(self.cb_empty)

    def cb_empty(self, res: List[Tuple[bool, object]]) -> None:
        self.callbackRan = 1
        self.assertEqual([], res)

    def tearDown(self) -> None:
        self.assertTrue(self.callbackRan, 'Callback was never run.')