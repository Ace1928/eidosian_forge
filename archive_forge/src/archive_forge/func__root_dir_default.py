import os
import shutil
from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_dir_exists
from tornado.web import HTTPError
from traitlets import Unicode
from jupyter_server import _tz as tz
from .checkpoints import (
from .fileio import AsyncFileManagerMixin, FileManagerMixin
def _root_dir_default(self):
    if not self.parent:
        return os.getcwd()
    return self.parent.root_dir