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
def _setupRaceState(numDeferreds: int) -> tuple[list[int], list[Deferred[object]]]:
    """
    Create a list of Deferreds and a corresponding list of integers
    tracking how many times each Deferred has been cancelled.  Without
    additional steps the Deferreds will never fire.
    """
    cancelledState = [0] * numDeferreds
    ds: list[Deferred[object]] = []
    for n in range(numDeferreds):

        def cancel(d: Deferred[object], n: int=n) -> None:
            cancelledState[n] += 1
        ds.append(Deferred(canceller=cancel))
    return (cancelledState, ds)