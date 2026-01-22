from __future__ import annotations
import functools
import os
import shutil
import stat
import sys
import re
import typing as T
from pathlib import Path
from . import mesonlib
from . import mlog
from .mesonlib import MachineChoice, OrderedSet
@staticmethod
@functools.lru_cache(maxsize=None)
def _windows_sanitize_path(path: str) -> str:
    if 'USERPROFILE' not in os.environ:
        return path
    appstore_dir = Path(os.environ['USERPROFILE']) / 'AppData' / 'Local' / 'Microsoft' / 'WindowsApps'
    paths = []
    for each in path.split(os.pathsep):
        if Path(each) != appstore_dir:
            paths.append(each)
        elif 'WindowsApps' in sys.executable:
            paths.append(os.path.dirname(sys.executable))
    return os.pathsep.join(paths)