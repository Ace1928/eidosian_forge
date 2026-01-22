import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class PersistentDictComponents(PersistentComponents, dict):
    pass