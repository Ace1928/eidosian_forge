import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_stressReferences(self):
    reref = []
    toplevelTuple = ({'list': reref}, reref)
    reref.append(toplevelTuple)
    s = jelly.jelly(toplevelTuple)
    z = jelly.unjelly(s)
    self.assertIs(z[0]['list'], z[1])
    self.assertIs(z[0]['list'][0], z)