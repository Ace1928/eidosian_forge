from functools import _make_key, wraps
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple, Type
def _get_lock(self, key) -> Any:
    with self._lock:
        if key not in self._locks:
            self._locks[key] = self._lock_type()
        return self._locks[key]