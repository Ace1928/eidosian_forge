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
def _handle_git_information_request(self) -> None:
    msg = ForwardMsg()
    try:
        from streamlit.git_util import GitRepo
        repo = GitRepo(self._script_data.main_script_path)
        repo_info = repo.get_repo_info()
        if repo_info is None:
            return
        repository_name, branch, module = repo_info
        if repository_name.endswith('.git'):
            repository_name = repository_name[:-4]
        msg.git_info_changed.repository = repository_name
        msg.git_info_changed.branch = branch
        msg.git_info_changed.module = module
        msg.git_info_changed.untracked_files[:] = repo.untracked_files
        msg.git_info_changed.uncommitted_files[:] = repo.uncommitted_files
        if repo.is_head_detached:
            msg.git_info_changed.state = GitInfo.GitStates.HEAD_DETACHED
        elif len(repo.ahead_commits) > 0:
            msg.git_info_changed.state = GitInfo.GitStates.AHEAD_OF_REMOTE
        else:
            msg.git_info_changed.state = GitInfo.GitStates.DEFAULT
        self._enqueue_forward_msg(msg)
    except Exception as ex:
        _LOGGER.debug('Obtaining Git information produced an error', exc_info=ex)