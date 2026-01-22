import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class TestControl:
    """Controls a running test run, allowing it to be interrupted.

    :ivar shouldStop: If True, tests should not run and should instead
        return immediately. Similarly a TestSuite should check this between
        each test and if set stop dispatching any new tests and return.
    """

    def __init__(self):
        super().__init__()
        self.shouldStop = False

    def stop(self):
        """Indicate that tests should stop running."""
        self.shouldStop = True