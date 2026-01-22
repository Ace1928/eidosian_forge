import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
class PersistentAdapterRegistry(VerifyingAdapterRegistry):

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in list(state):
            if k in self._delegated or k.startswith('_v'):
                state.pop(k)
        state.pop('ro', None)
        return state

    def __setstate__(self, state):
        bases = state.pop('__bases__', ())
        self.__dict__.update(state)
        self._createLookup()
        self.__bases__ = bases
        self._v_lookup.changed(self)