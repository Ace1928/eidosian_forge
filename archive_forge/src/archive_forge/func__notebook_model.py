from __future__ import annotations
import errno
import math
import mimetypes
import os
import platform
import shutil
import stat
import subprocess
import sys
import typing as t
import warnings
from datetime import datetime
from pathlib import Path
import nbformat
from anyio.to_thread import run_sync
from jupyter_core.paths import exists, is_file_hidden, is_hidden
from send2trash import send2trash
from tornado import web
from traitlets import Bool, Int, TraitError, Unicode, default, validate
from jupyter_server import _tz as tz
from jupyter_server.base.handlers import AuthenticatedFileHandler
from jupyter_server.transutils import _i18n
from jupyter_server.utils import to_api_path
from .filecheckpoints import AsyncFileCheckpoints, FileCheckpoints
from .fileio import AsyncFileManagerMixin, FileManagerMixin
from .manager import AsyncContentsManager, ContentsManager, copy_pat
def _notebook_model(self, path, content=True, require_hash=False):
    """Build a notebook model

        if content is requested, the notebook content will be populated
        as a JSON structure (not double-serialized)

        if require_hash is true, the model will include 'hash'
        """
    model = self._base_model(path)
    model['type'] = 'notebook'
    os_path = self._get_os_path(path)
    bytes_content = None
    if content:
        validation_error: dict[str, t.Any] = {}
        nb, bytes_content = self._read_notebook(os_path, as_version=4, capture_validation_error=validation_error, raw=True)
        self.mark_trusted_cells(nb, path)
        model['content'] = nb
        model['format'] = 'json'
        self.validate_notebook_model(model, validation_error)
    if require_hash:
        if bytes_content is None:
            bytes_content, _ = self._read_file(os_path, 'byte')
        model.update(**self._get_hash(bytes_content))
    return model