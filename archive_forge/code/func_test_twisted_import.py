import sys
from io import StringIO
from twisted.internet import defer, reactor
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import TestCase
def test_twisted_import(self):
    """Importing twisted.__main__ does not execute twist."""
    output = StringIO()
    monkey = self.patch(sys, 'stdout', output)
    import twisted.__main__
    self.assertTrue(twisted.__main__)
    monkey.restore()
    self.assertEqual(output.getvalue(), '')