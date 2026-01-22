import threading
from typing import Any
from typing import Callable
from typing import MutableMapping
import weakref
def _sync_get(self, identifier: str, *args: Any, **kw: Any) -> Any:
    self._mutex.acquire()
    try:
        try:
            if identifier in self._values:
                return self._values[identifier]
            else:
                self._values[identifier] = value = self.creator(identifier, *args, **kw)
                return value
        except KeyError:
            self._values[identifier] = value = self.creator(identifier, *args, **kw)
            return value
    finally:
        self._mutex.release()