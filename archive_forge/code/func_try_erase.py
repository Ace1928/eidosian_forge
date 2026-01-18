import asyncio
import threading
from typing import Generic, TypeVar, Callable, Dict, Awaitable
def try_erase(self, key: _Key, client: _Client):
    with self._lock:
        if key not in self._live_clients:
            return
        current_client = self._live_clients[key]
        if current_client is not client:
            return
        del self._live_clients[key]
    self._closer(client)