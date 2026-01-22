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
class AsyncObjects(NamedTuple):
    """Container for all asyncio objects that Runtime manages.
    These cannot be initialized until the Runtime's eventloop is assigned.
    """
    eventloop: asyncio.AbstractEventLoop
    must_stop: asyncio.Event
    has_connection: asyncio.Event
    need_send_data: asyncio.Event
    started: asyncio.Future[None]
    stopped: asyncio.Future[None]