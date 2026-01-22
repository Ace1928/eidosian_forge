from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def _registerAdapterForClassOrInterface(self, original):
    """
        Register an adapter with L{components.registerAdapter} for the given
        class or interface and verify that the adapter can be looked up with
        L{components.getAdapterFactory}.
        """
    adapter = lambda o: None
    components.registerAdapter(adapter, original, ITest)
    self.assertIs(components.getAdapterFactory(original, ITest, None), adapter)