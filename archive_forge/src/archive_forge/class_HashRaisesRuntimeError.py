from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
class HashRaisesRuntimeError:
    """
    Things that don't hash (raise an Exception) should be ignored by the
    rebuilder.

    @ivar hashCalled: C{bool} set to True when __hash__ is called.
    """

    def __init__(self) -> None:
        self.hashCalled = False

    def __hash__(self) -> NoReturn:
        self.hashCalled = True
        raise RuntimeError('not a TypeError!')