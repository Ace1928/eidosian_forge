import collections
from collections.abc import Mapping as collections_Mapping
from pyomo.common.autoslots import AutoSlots
class DefaultComponentMap(ComponentMap):
    """A :py:class:`defaultdict` admitting Pyomo Components as keys

    This class is a replacement for defaultdict that allows Pyomo
    modeling components to be used as entry keys. The base
    implementation builds on :py:class:`ComponentMap`.

    """
    __slots__ = ('default_factory',)

    def __init__(self, default_factory=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = ans = self.default_factory()
        return ans

    def __getitem__(self, obj):
        _key = _hasher[obj.__class__](obj)
        if _key in self._dict:
            return self._dict[_key][1]
        else:
            return self.__missing__(obj)