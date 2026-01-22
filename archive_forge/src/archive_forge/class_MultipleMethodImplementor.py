from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
class MultipleMethodImplementor:
    """
    A precise implementation of L{IMultipleMethods}.
    """

    def methodOne(self):
        """
        @return: 1
        """
        return 1

    def methodTwo(self):
        """
        @return: 2
        """
        return 2