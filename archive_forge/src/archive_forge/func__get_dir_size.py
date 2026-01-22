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
def _get_dir_size(self, path='.'):
    """
        calls the command line program du to get the directory size
        """
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(['du', '-sk', path], capture_output=True, check=True).stdout.split()
        else:
            result = subprocess.run(['du', '-s', '--block-size=1', path], capture_output=True, check=True).stdout.split()
        self.log.info(f'current status of du command {result}')
        size = result[0].decode('utf-8')
    except Exception:
        self.log.warning('Not able to get the size of the %s directory. Copying might be slow if the directory is large!', path)
        return '0'
    return size