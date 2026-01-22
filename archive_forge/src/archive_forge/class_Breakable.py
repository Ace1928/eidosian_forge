import os
import weakref
from collections import deque
from twisted.python import reflect
from twisted.python.reflect import (
from twisted.trial.unittest import SynchronousTestCase as TestCase
class Breakable:
    breakRepr = False
    breakStr = False

    def __str__(self) -> str:
        if self.breakStr:
            raise RuntimeError('str!')
        else:
            return '<Breakable>'

    def __repr__(self) -> str:
        if self.breakRepr:
            raise RuntimeError('repr!')
        else:
            return 'Breakable()'