from __future__ import annotations
import errno
import hashlib
import os
import shutil
from base64 import decodebytes, encodebytes
from contextlib import contextmanager
from functools import partial
import nbformat
from anyio.to_thread import run_sync
from tornado.web import HTTPError
from traitlets import Bool, Enum
from traitlets.config import Configurable
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.utils import ApiPath, to_api_path, to_os_path
def _save_notebook(self, os_path, nb, capture_validation_error=None):
    """Save a notebook to an os_path."""
    with self.atomic_writing(os_path, encoding='utf-8') as f:
        nbformat.write(nb, f, version=nbformat.NO_CONVERT, capture_validation_error=capture_validation_error)