from __future__ import annotations
import gc
import sys
import threading
import types
from contextlib import contextmanager
from enum import Enum
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Callable, Final
from blinker import Signal
from streamlit import config, runtime, source_util, util
from streamlit.error_util import handle_uncaught_app_exception
from streamlit.logger import get_logger
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.scriptrunner.script_requests import (
from streamlit.runtime.scriptrunner.script_run_context import (
from streamlit.runtime.state import (
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.vendor.ipython.modified_sys_path import modified_sys_path
def _enqueue_forward_msg(self, msg: ForwardMsg) -> None:
    """Enqueue a ForwardMsg to our browser queue.
        This private function is called by ScriptRunContext only.

        It may be called from the script thread OR the main thread.
        """
    if not config.get_option('runner.installTracer'):
        self._maybe_handle_execution_control_request()
    self.on_event.send(self, event=ScriptRunnerEvent.ENQUEUE_FORWARD_MSG, forward_msg=msg)