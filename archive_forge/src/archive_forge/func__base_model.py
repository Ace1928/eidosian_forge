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
def _base_model(self, path):
    """Build the common base of a contents model"""
    os_path = self._get_os_path(path)
    info = os.lstat(os_path)
    four_o_four = 'file or directory does not exist: %r' % path
    if not self.allow_hidden and is_hidden(os_path, self.root_dir):
        self.log.info('Refusing to serve hidden file or directory %r, via 404 Error', os_path)
        raise web.HTTPError(404, four_o_four)
    try:
        size = info.st_size
    except (ValueError, OSError):
        self.log.warning('Unable to get size.')
        size = None
    try:
        last_modified = tz.utcfromtimestamp(info.st_mtime)
    except (ValueError, OSError):
        self.log.warning('Invalid mtime %s for %s', info.st_mtime, os_path)
        last_modified = datetime(1970, 1, 1, 0, 0, tzinfo=tz.UTC)
    try:
        created = tz.utcfromtimestamp(info.st_ctime)
    except (ValueError, OSError):
        self.log.warning('Invalid ctime %s for %s', info.st_ctime, os_path)
        created = datetime(1970, 1, 1, 0, 0, tzinfo=tz.UTC)
    model = {}
    model['name'] = path.rsplit('/', 1)[-1]
    model['path'] = path
    model['last_modified'] = last_modified
    model['created'] = created
    model['content'] = None
    model['format'] = None
    model['mimetype'] = None
    model['size'] = size
    model['writable'] = self.is_writable(path)
    model['hash'] = None
    model['hash_algorithm'] = None
    return model