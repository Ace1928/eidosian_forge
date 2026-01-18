from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_proxyAttribute(self):
    """
        Proxy objects should proxy declared attributes, but not other
        attributes.
        """
    yayable = Yayable()
    yayable.ifaceAttribute = object()
    proxy = proxyForInterface(IProxiedInterface)(yayable)
    self.assertIs(proxy.ifaceAttribute, yayable.ifaceAttribute)
    self.assertRaises(AttributeError, lambda: proxy.yays)