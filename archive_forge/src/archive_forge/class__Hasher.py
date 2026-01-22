import collections
from collections.abc import Mapping as collections_Mapping
from pyomo.common.autoslots import AutoSlots
class _Hasher(collections.defaultdict):

    def __init__(self, *args, **kwargs):
        super().__init__(lambda: self._missing_impl, *args, **kwargs)
        self[tuple] = self._tuple

    def _missing_impl(self, val):
        try:
            hash(val)
            self[val.__class__] = self._hashable
        except:
            self[val.__class__] = self._unhashable
        return self[val.__class__](val)

    @staticmethod
    def _hashable(val):
        return val

    @staticmethod
    def _unhashable(val):
        return id(val)

    def _tuple(self, val):
        return tuple((self[i.__class__](i) for i in val))