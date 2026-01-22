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
def _search_dir(self, name: str, search_dir: T.Optional[str]) -> T.Optional[list]:
    if search_dir is None:
        return None
    trial = os.path.join(search_dir, name)
    if os.path.exists(trial):
        if self._is_executable(trial):
            return [trial]
        return self._shebang_to_cmd(trial)
    elif mesonlib.is_windows():
        for ext in self.windows_exts:
            trial_ext = f'{trial}.{ext}'
            if os.path.exists(trial_ext):
                return [trial_ext]
    return None