import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class InterfaceClassSubclass(InterfaceClass):
    pass