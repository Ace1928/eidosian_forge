import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
def _find_git_in_win_path():
    for exe in ('git.exe', 'git.cmd'):
        for path in os.environ.get('PATH', '').split(';'):
            if os.path.exists(os.path.join(path, exe)):
                git_dir, _bin_dir = os.path.split(path)
                yield git_dir
                parent_dir, basename = os.path.split(git_dir)
                if basename == 'mingw32' or basename == 'mingw64':
                    yield parent_dir
                break