import collections
from typing import Any, Set
import weakref
@staticmethod
def _from_storage(storage):
    result = ObjectIdentitySet()
    result._storage = storage
    return result