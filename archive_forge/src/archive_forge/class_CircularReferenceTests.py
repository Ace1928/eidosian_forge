import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class CircularReferenceTests(TestCase):
    """
    Tests for circular references handling in the jelly/unjelly process.
    """

    def test_simpleCircle(self):
        jelly.setUnjellyableForClass(ClassA, ClassA)
        jelly.setUnjellyableForClass(ClassB, ClassB)
        a = jelly.unjelly(jelly.jelly(ClassA()))
        self.assertIs(a.ref.ref, a, 'Identity not preserved in circular reference')

    def test_circleWithInvoker(self):

        class DummyInvokerClass:
            pass
        dummyInvoker = DummyInvokerClass()
        dummyInvoker.serializingPerspective = None
        a0 = ClassA()
        jelly.setUnjellyableForClass(ClassA, ClassA)
        jelly.setUnjellyableForClass(ClassB, ClassB)
        j = jelly.jelly(a0, invoker=dummyInvoker)
        a1 = jelly.unjelly(j)
        self.failUnlessIdentical(a1.ref.ref, a1, 'Identity not preserved in circular reference')

    def test_set(self):
        """
        Check that a C{set} can contain a circular reference and be serialized
        and unserialized without losing the reference.
        """
        s = set()
        a = SimpleJellyTest(s, None)
        s.add(a)
        res = jelly.unjelly(jelly.jelly(a))
        self.assertIsInstance(res.x, set)
        self.assertEqual(list(res.x), [res])

    def test_frozenset(self):
        """
        Check that a L{frozenset} can contain a circular reference and be
        serialized and unserialized without losing the reference.
        """
        a = SimpleJellyTest(None, None)
        s = frozenset([a])
        a.x = s
        res = jelly.unjelly(jelly.jelly(a))
        self.assertIsInstance(res.x, frozenset)
        self.assertEqual(list(res.x), [res])