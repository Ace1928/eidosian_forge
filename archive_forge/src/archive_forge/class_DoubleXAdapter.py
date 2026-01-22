from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
@comparable
class DoubleXAdapter:
    """
    Adapter with __cmp__.
    """
    num = 42

    def __init__(self, original):
        self.original = original

    def xx(self):
        return (self.original.x(), self.original.x())

    def __cmp__(self, other):
        return cmp(self.num, other.num)