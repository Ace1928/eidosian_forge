from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, cast
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
def to_active(self) -> ActiveSessionInfo:
    assert self.is_active(), 'A SessionInfo with no client cannot be active!'
    return cast(ActiveSessionInfo, self)