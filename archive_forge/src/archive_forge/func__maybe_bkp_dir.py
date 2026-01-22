import os
import shutil
from contextlib import contextmanager
from distutils import log
from distutils.core import Command
from pathlib import Path
from .. import _normalization
@contextmanager
def _maybe_bkp_dir(self, dir_path: str, requires_bkp: bool):
    if requires_bkp:
        bkp_name = f'{dir_path}.__bkp__'
        _rm(bkp_name, ignore_errors=True)
        shutil.copytree(dir_path, bkp_name, dirs_exist_ok=True, symlinks=True)
        try:
            yield
        finally:
            _rm(dir_path, ignore_errors=True)
            shutil.move(bkp_name, dir_path)
    else:
        yield