from __future__ import annotations
import asyncio
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Callable, Final
import streamlit.elements.exception as exception_utils
from streamlit import config, runtime, source_util
from streamlit.case_converters import to_snake_case
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.Common_pb2 import FileURLs, FileURLsRequest
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.GitInfo_pb2 import GitInfo
from streamlit.proto.NewSession_pb2 import (
from streamlit.proto.PagesChanged_pb2 import PagesChanged
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.fragment import FragmentStorage, MemoryFragmentStorage
from streamlit.runtime.metrics_util import Installation
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner import RerunData, ScriptRunner, ScriptRunnerEvent
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.secrets import secrets_singleton
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.version import STREAMLIT_VERSION_STRING
from streamlit.watcher import LocalSourcesWatcher
def _create_new_session_message(self, page_script_hash: str, fragment_ids_this_run: set[str] | None=None) -> ForwardMsg:
    """Create and return a new_session ForwardMsg."""
    msg = ForwardMsg()
    msg.new_session.script_run_id = _generate_scriptrun_id()
    msg.new_session.name = self._script_data.name
    msg.new_session.main_script_path = self._script_data.main_script_path
    msg.new_session.page_script_hash = page_script_hash
    if fragment_ids_this_run:
        msg.new_session.fragment_ids_this_run.extend(fragment_ids_this_run)
    _populate_app_pages(msg.new_session, self._script_data.main_script_path)
    _populate_config_msg(msg.new_session.config)
    _populate_theme_msg(msg.new_session.custom_theme)
    imsg = msg.new_session.initialize
    _populate_user_info_msg(imsg.user_info)
    imsg.environment_info.streamlit_version = STREAMLIT_VERSION_STRING
    imsg.environment_info.python_version = '.'.join(map(str, sys.version_info))
    imsg.session_status.run_on_save = self._run_on_save
    imsg.session_status.script_is_running = self._state == AppSessionState.APP_IS_RUNNING
    imsg.is_hello = self._script_data.is_hello
    imsg.session_id = self.id
    return msg