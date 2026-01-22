from __future__ import annotations
import io
import json
from email.parser import Parser
from importlib.resources import files
from typing import TYPE_CHECKING, Any
import js  # type: ignore[import-not-found]
from pyodide.ffi import (  # type: ignore[import-not-found]
from .request import EmscriptenRequest
from .response import EmscriptenResponse
def _show_streaming_warning() -> None:
    global _SHOWN_STREAMING_WARNING
    if not _SHOWN_STREAMING_WARNING:
        _SHOWN_STREAMING_WARNING = True
        message = "Can't stream HTTP requests because: \n"
        if not is_cross_origin_isolated():
            message += '  Page is not cross-origin isolated\n'
        if is_in_browser_main_thread():
            message += '  Python is running in main browser thread\n'
        if not is_worker_available():
            message += ' Worker or Blob classes are not available in this environment.'
        if streaming_ready() is False:
            message += " Streaming fetch worker isn't ready. If you want to be sure that streaming fetch\nis working, you need to call: 'await urllib3.contrib.emscripten.fetch.wait_for_streaming_ready()`"
        from js import console
        console.warn(message)