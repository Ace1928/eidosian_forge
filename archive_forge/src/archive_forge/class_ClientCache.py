import threading
from typing import Generic, TypeVar, Callable, Optional
class ClientCache(Generic[_Client]):
    _ClientFactory = Callable[[], _Client]
    _factory: _ClientFactory
    _latest: Optional[_Client]
    _remaining_uses: int
    _lock: threading.Lock

    def __init__(self, factory: _ClientFactory):
        self._factory = factory
        self._latest = None
        self._remaining_uses = 0
        self._lock = threading.Lock()

    def get(self) -> _Client:
        with self._lock:
            if self._remaining_uses <= 0:
                self._remaining_uses = _MAX_CLIENT_USES
                self._latest = self._factory()
            self._remaining_uses -= 1
            return self._latest