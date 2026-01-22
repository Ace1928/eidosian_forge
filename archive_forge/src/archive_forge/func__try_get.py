from functools import _make_key, wraps
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple, Type
def _try_get(self, key) -> Tuple[bool, Any]:
    with self._lock:
        if key in self._store:
            return (True, self._store[key])
        return (False, None)