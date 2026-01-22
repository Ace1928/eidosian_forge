from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
@implementer(IMeta)
class ComponentAdder(components.Adapter):
    """
    Adder for componentized adapter tests.
    """

    def __init__(self, original):
        components.Adapter.__init__(self, original)
        self.num = self.original.num

    def add(self, num):
        self.num += num
        return self.num