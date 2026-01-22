from __future__ import annotations
import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath
from ._internal import _log
def _find_watchdog_paths(extra_files: set[str], exclude_patterns: set[str]) -> t.Iterable[str]:
    """Find paths for the stat reloader to watch. Looks at the same
    sources as the stat reloader, but watches everything under
    directories instead of individual files.
    """
    dirs = set()
    for name in chain(list(sys.path), extra_files):
        name = os.path.abspath(name)
        if os.path.isfile(name):
            name = os.path.dirname(name)
        dirs.add(name)
    for name in _iter_module_paths():
        dirs.add(os.path.dirname(name))
    _remove_by_pattern(dirs, exclude_patterns)
    return _find_common_roots(dirs)