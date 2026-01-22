import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
class MathTests(TestCase):

    def test_int2b128(self):
        funkylist = list(range(0, 100)) + list(range(1000, 1100)) + list(range(1000000, 1000100)) + [1024 ** 10]
        for i in funkylist:
            x = BytesIO()
            banana.int2b128(i, x.write)
            v = x.getvalue()
            y = banana.b1282int(v)
            self.assertEqual(y, i)