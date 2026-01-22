from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def _subclassAdapterRegistrationForClassOrInterface(self, original):
    """
        Verify that a new adapter can be registered for a particular
        to-interface from a subclass of a type or interface which already has an
        adapter registered to that interface and that the subclass adapter takes
        precedence over the base class adapter.
        """
    firstAdapter = lambda o: True
    secondAdapter = lambda o: False

    class TheSubclass(original):
        pass
    components.registerAdapter(firstAdapter, original, ITest)
    components.registerAdapter(secondAdapter, TheSubclass, ITest)
    self.assertIs(components.getAdapterFactory(original, ITest, None), firstAdapter)
    self.assertIs(components.getAdapterFactory(TheSubclass, ITest, None), secondAdapter)