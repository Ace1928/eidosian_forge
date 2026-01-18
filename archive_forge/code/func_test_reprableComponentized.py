from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_reprableComponentized(self):
    """
        C{ReprableComponentized} has a C{__repr__} that lists its cache.
        """
    rc = components.ReprableComponentized()
    rc.setComponent(ITest, 'hello')
    result = repr(rc)
    self.assertIn('ITest', result)
    self.assertIn('hello', result)