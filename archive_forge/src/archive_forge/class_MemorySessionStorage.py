from __future__ import annotations
from typing import MutableMapping
from cachetools import TTLCache
from streamlit.runtime.session_manager import SessionInfo, SessionStorage
class MemorySessionStorage(SessionStorage):
    """A SessionStorage that stores sessions in memory.

    At most maxsize sessions are stored with a TTL of ttl seconds. This class is really
    just a thin wrapper around cachetools.TTLCache that complies with the SessionStorage
    protocol.
    """

    def __init__(self, maxsize: int=128, ttl_seconds: int=2 * 60) -> None:
        """Instantiate a new MemorySessionStorage.

        Parameters
        ----------
        maxsize
            The maximum number of sessions we allow to be stored in this
            MemorySessionStorage. If an entry needs to be removed because we have
            exceeded this number, either
              * an expired entry is removed, or
              * the least recently used entry is removed (if no entries have expired).

        ttl_seconds
            The time in seconds for an entry added to a MemorySessionStorage to live.
            After this amount of time has passed for a given entry, it becomes
            inaccessible and will be removed eventually.
        """
        self._cache: MutableMapping[str, SessionInfo] = TTLCache(maxsize=maxsize, ttl=ttl_seconds)

    def get(self, session_id: str) -> SessionInfo | None:
        return self._cache.get(session_id, None)

    def save(self, session_info: SessionInfo) -> None:
        self._cache[session_info.session.id] = session_info

    def delete(self, session_id: str) -> None:
        del self._cache[session_id]

    def list(self) -> list[SessionInfo]:
        return list(self._cache.values())