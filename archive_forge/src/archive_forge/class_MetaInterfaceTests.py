from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class MetaInterfaceTests(RegistryUsingMixin, unittest.SynchronousTestCase):

    def test_basic(self):
        """
        Registered adapters can be used to adapt classes to an interface.
        """
        components.registerAdapter(MetaAdder, MetaNumber, IMeta)
        n = MetaNumber(1)
        self.assertEqual(IMeta(n).add(1), 2)

    def testComponentizedInteraction(self):
        components.registerAdapter(ComponentAdder, ComponentNumber, IMeta)
        c = ComponentNumber()
        IMeta(c).add(1)
        IMeta(c).add(1)
        self.assertEqual(IMeta(c).add(1), 3)

    def testAdapterWithCmp(self):
        components.registerAdapter(DoubleXAdapter, IAttrX, IAttrXX)
        xx = IAttrXX(Xcellent())
        self.assertEqual(('x!', 'x!'), xx.xx())