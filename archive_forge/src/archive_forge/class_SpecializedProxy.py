from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class SpecializedProxy(proxyForInterface(IProxiedInterface)):
    """
            A specialized proxy which can decrement the number of yays.
            """

    def boo(self):
        """
                Decrement the number of yays.
                """
        self.original.yays -= 1