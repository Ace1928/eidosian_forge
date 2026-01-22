from __future__ import annotations
import os
from fnmatch import fnmatch
from typing import (
import param
from ..io import PeriodicCallback
from ..layout import (
from ..util import fullpath
from ..viewable import Layoutable
from .base import CompositeWidget
from .button import Button
from .input import TextInput
from .select import CrossSelector
def _scan_path(path: str, file_pattern='*') -> Tuple[List[str], List[str]]:
    """
    Scans the supplied path for files and directories and optionally
    filters the files with the file keyword, returning a list of sorted
    paths of all directories and files.

    Arguments
    ---------
    path: str
        The path to search
    file_pattern: str
        A glob-like pattern to filter the files

    Returns
    -------
    A sorted list of directory paths, A sorted list of files
    """
    paths = [os.path.join(path, p) for p in os.listdir(path)]
    dirs = [p for p in paths if os.path.isdir(p)]
    files = [p for p in paths if os.path.isfile(p) and fnmatch(os.path.basename(p), file_pattern)]
    for p in paths:
        if not os.path.islink(p):
            continue
        path = os.path.realpath(p)
        if os.path.isdir(path):
            dirs.append(p)
        elif os.path.isfile(path):
            dirs.append(p)
        else:
            continue
    return (dirs, files)