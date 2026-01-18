import unittest
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface import classImplementsOnly
from zope.interface import directlyProvidedBy
from zope.interface import directlyProvides
from zope.interface import implementedBy
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface.tests import odd
def test_odd_metaclass_that_doesnt_subclass_type(self):

    class A:
        a = 1
    A = odd.MetaClass('A', A.__bases__, A.__dict__)

    class B:
        b = 1
    B = odd.MetaClass('B', B.__bases__, B.__dict__)

    class C(A, B):
        pass
    self.assertEqual(C.__bases__, (A, B))
    a = A()
    aa = A()
    self.assertEqual(a.a, 1)
    self.assertEqual(aa.a, 1)
    aa.a = 2
    self.assertEqual(a.a, 1)
    self.assertEqual(aa.a, 2)
    c = C()
    self.assertEqual(c.a, 1)
    self.assertEqual(c.b, 1)
    c.b = 2
    self.assertEqual(c.b, 2)
    C.c = 1
    self.assertEqual(c.c, 1)
    c.c
    self.assertIs(C.__class__.__class__, C.__class__)