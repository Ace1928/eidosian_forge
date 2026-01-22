import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class _Factory2:

    def __init__(self, context1, context2):
        self.context = (context1, context2)