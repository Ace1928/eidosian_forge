import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class Summer:
    """
    A class we look up as part of the LookupsTests.
    """

    def reallySet(self):
        """
        Do something.
        """