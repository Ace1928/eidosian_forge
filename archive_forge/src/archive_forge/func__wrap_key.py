import collections
from typing import Any, Set
import weakref
def _wrap_key(self, key):
    return _WeakObjectIdentityWrapper(key)