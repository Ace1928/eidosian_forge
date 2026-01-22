import copy
import weakref
from pyomo.common.autoslots import AutoSlots
def _clear_parent_and_storage_key(self):
    object.__setattr__(self, '_parent', None)
    object.__setattr__(self, '_storage_key', None)