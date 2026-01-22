import errno
import os
import sys
from typing import Optional
from twisted.trial.unittest import TestCase
class FakeResourceModule:
    """
    Fake version of L{resource} which hard-codes a particular rlimit for maximum
    open files.

    @ivar _limit: The value to return for the hard limit of number of open files.
    """
    RLIMIT_NOFILE = 1

    def __init__(self, limit):
        self._limit = limit

    def getrlimit(self, no):
        """
        A fake of L{resource.getrlimit} which returns a pre-determined result.
        """
        if no == self.RLIMIT_NOFILE:
            return [0, self._limit]
        return [123, 456]