import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def _check_newstyle(self, a, b):
    self.assertEqual(a.id, b.id)
    self.assertEqual(a.classAttr, 4)
    self.assertEqual(b.classAttr, 4)
    self.assertEqual(len(a.children), len(b.children))
    for x, y in zip(a.children, b.children):
        self._check_newstyle(x, y)