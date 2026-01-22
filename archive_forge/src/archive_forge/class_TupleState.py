import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class TupleState:

    def __init__(self, other):
        self.other = other

    def __getstate__(self):
        return (self.other,)

    def __setstate__(self, state):
        self.other = state[0]

    def __hash__(self):
        return hash(self.other)