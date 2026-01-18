import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_simpleCircle(self):
    jelly.setUnjellyableForClass(ClassA, ClassA)
    jelly.setUnjellyableForClass(ClassB, ClassB)
    a = jelly.unjelly(jelly.jelly(ClassA()))
    self.assertIs(a.ref.ref, a, 'Identity not preserved in circular reference')