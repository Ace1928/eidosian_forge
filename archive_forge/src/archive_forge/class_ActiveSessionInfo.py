from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, cast
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
@dataclass
class ActiveSessionInfo:
    """Type containing data related to an active session.

    This type is nearly identical to SessionInfo. The difference is that when using it,
    we are guaranteed that SessionClient is not None.
    """
    client: SessionClient
    session: AppSession
    script_run_count: int = 0