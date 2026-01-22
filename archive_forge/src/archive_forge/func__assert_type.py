import collections
from typing import Any, Set
import weakref
def _assert_type(self, other):
    if not isinstance(other, _ObjectIdentityWrapper):
        raise TypeError('Cannot compare wrapped object with unwrapped object')