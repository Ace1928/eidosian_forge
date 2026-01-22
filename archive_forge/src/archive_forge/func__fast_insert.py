import collections.abc
from pyomo.core.kernel.homogeneous_container import IHomogeneousContainer
def _fast_insert(self, i, item):
    item._update_parent_and_storage_key(self, i)
    self._data.insert(i, item)