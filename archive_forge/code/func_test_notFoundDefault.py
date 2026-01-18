from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_notFoundDefault(self) -> None:
    """
        If none of the objects passed in the list to L{acquireAttribute} have
        the requested attribute and a default value is given, the default value
        is returned.
        """
    default = object()
    self.assertTrue(default is acquireAttribute([object()], 'foo', default))