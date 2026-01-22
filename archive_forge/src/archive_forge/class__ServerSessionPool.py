from the same snapshot timestamp. The server chooses the latest
from __future__ import annotations
import collections
import time
import uuid
from collections.abc import Mapping as _Mapping
from typing import (
from bson.binary import Binary
from bson.int64 import Int64
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot
from pymongo.cursor import _ConnectionManager
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_type import SERVER_TYPE
from pymongo.write_concern import WriteConcern
class _ServerSessionPool(collections.deque):
    """Pool of _ServerSession objects.

    This class is not thread-safe, access it while holding the Topology lock.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.generation = 0

    def reset(self) -> None:
        self.generation += 1
        self.clear()

    def pop_all(self) -> list[_ServerSession]:
        ids = []
        while self:
            ids.append(self.pop().session_id)
        return ids

    def get_server_session(self, session_timeout_minutes: float) -> _ServerSession:
        self._clear_stale(session_timeout_minutes)
        while self:
            s = self.popleft()
            if not s.timed_out(session_timeout_minutes):
                return s
        return _ServerSession(self.generation)

    def return_server_session(self, server_session: _ServerSession, session_timeout_minutes: Optional[float]) -> None:
        if session_timeout_minutes is not None:
            self._clear_stale(session_timeout_minutes)
            if server_session.timed_out(session_timeout_minutes):
                return
        self.return_server_session_no_lock(server_session)

    def return_server_session_no_lock(self, server_session: _ServerSession) -> None:
        if server_session.generation == self.generation and (not server_session.dirty):
            self.appendleft(server_session)

    def _clear_stale(self, session_timeout_minutes: float) -> None:
        while self:
            if self[-1].timed_out(session_timeout_minutes):
                self.pop()
            else:
                break