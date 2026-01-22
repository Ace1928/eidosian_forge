import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class E:
    """
    Dummy new-style class with slots.
    """
    __slots__ = ('x', 'y')

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def __getstate__(self):
        return {'x': self.x, 'y': self.y}

    def __setstate__(self, state):
        self.x = state['x']
        self.y = state['y']