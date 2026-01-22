import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class PersistentComponents(Components):

    def _init_registries(self):
        self.adapters = PersistentAdapterRegistry()
        self.utilities = PersistentAdapterRegistry()