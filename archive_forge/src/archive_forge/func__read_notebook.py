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
def _read_notebook(self, os_path, as_version=4, capture_validation_error=None, raw: bool=False):
    """Read a notebook from an os path."""
    answer = self._read_file(os_path, 'text', raw=raw)
    try:
        nb = nbformat.reads(answer[0], as_version=as_version, capture_validation_error=capture_validation_error)
        return (nb, answer[2]) if raw else nb
    except Exception as e:
        e_orig = e
    tmp_path = path_to_intermediate(os_path)
    if not self.use_atomic_writing or not os.path.exists(tmp_path):
        raise HTTPError(400, f'Unreadable Notebook: {os_path} {e_orig!r}')
    invalid_file = path_to_invalid(os_path)
    replace_file(os_path, invalid_file)
    replace_file(tmp_path, os_path)
    return self._read_notebook(os_path, as_version, capture_validation_error=capture_validation_error, raw=raw)