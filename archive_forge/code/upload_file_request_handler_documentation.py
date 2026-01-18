from __future__ import annotations
from typing import Any, Callable
import tornado.httputil
import tornado.web
from streamlit import config
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.uploaded_file_manager import UploadedFileRec
from streamlit.web.server import routes, server_util
Delete file request handler.