from __future__ import annotations
import collections
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Counter, Dict, Final, Union
from urllib import parse
from typing_extensions import TypeAlias
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Command
from streamlit.runtime.scriptrunner.script_requests import ScriptRequests
from streamlit.runtime.state import SafeSessionState
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
import streamlit
def add_script_run_ctx(thread: threading.Thread | None=None, ctx: ScriptRunContext | None=None):
    """Adds the current ScriptRunContext to a newly-created thread.

    This should be called from this thread's parent thread,
    before the new thread starts.

    Parameters
    ----------
    thread : threading.Thread
        The thread to attach the current ScriptRunContext to.
    ctx : ScriptRunContext or None
        The ScriptRunContext to add, or None to use the current thread's
        ScriptRunContext.

    Returns
    -------
    threading.Thread
        The same thread that was passed in, for chaining.

    """
    if thread is None:
        thread = threading.current_thread()
    if ctx is None:
        ctx = get_script_run_ctx()
    if ctx is not None:
        setattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME, ctx)
    return thread