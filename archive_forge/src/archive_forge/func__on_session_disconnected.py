from __future__ import annotations
import asyncio
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Final, NamedTuple
from streamlit import config
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.caching import (
from streamlit.runtime.caching.storage.local_disk_cache_storage import (
from streamlit.runtime.forward_msg_cache import (
from streamlit.runtime.legacy_caching.caching import _mem_caches
from streamlit.runtime.media_file_manager import MediaFileManager
from streamlit.runtime.media_file_storage import MediaFileStorage
from streamlit.runtime.memory_session_storage import MemorySessionStorage
from streamlit.runtime.runtime_util import is_cacheable_msg
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.session_manager import (
from streamlit.runtime.state import (
from streamlit.runtime.stats import StatsManager
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.runtime.websocket_session_manager import WebsocketSessionManager
from streamlit.watcher import LocalSourcesWatcher
def _on_session_disconnected(self) -> None:
    """Set the runtime state to NO_SESSIONS_CONNECTED if the last active
        session was disconnected.
        """
    if self._state == RuntimeState.ONE_OR_MORE_SESSIONS_CONNECTED and self._session_mgr.num_active_sessions() == 0:
        self._get_async_objs().has_connection.clear()
        self._set_state(RuntimeState.NO_SESSIONS_CONNECTED)