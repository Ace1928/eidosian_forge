from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_proxyInheritance(self):
    """
        Subclasses of the class returned from L{proxyForInterface} should be
        able to upcall methods by reference to their superclass, as any normal
        Python class can.
        """

    class YayableWrapper(proxyForInterface(IProxiedInterface)):
        """
            This class does not override any functionality.
            """

    class EnhancedWrapper(YayableWrapper):
        """
            This class overrides the 'yay' method.
            """
        wrappedYays = 1

        def yay(self, *a, **k):
            self.wrappedYays += 1
            return YayableWrapper.yay(self, *a, **k) + 7
    yayable = Yayable()
    wrapper = EnhancedWrapper(yayable)
    self.assertEqual(wrapper.yay(3, 4, x=5, y=6), 8)
    self.assertEqual(yayable.yayArgs, [((3, 4), dict(x=5, y=6))])