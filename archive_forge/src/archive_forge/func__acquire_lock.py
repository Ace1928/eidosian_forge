import abc
import configparser as cp
import fnmatch
from functools import wraps
import inspect
from io import BufferedReader, IOBase
import logging
import os
import os.path as osp
import re
import sys
from git.compat import defenc, force_text
from git.util import LockFile
from typing import (
from git.types import Lit_config_levels, ConfigLevels_Tup, PathLike, assert_never, _T
def _acquire_lock(self) -> None:
    if not self._read_only:
        if not self._lock:
            if isinstance(self._file_or_files, (str, os.PathLike)):
                file_or_files = self._file_or_files
            elif isinstance(self._file_or_files, (tuple, list, Sequence)):
                raise ValueError('Write-ConfigParsers can operate on a single file only, multiple files have been passed')
            else:
                file_or_files = self._file_or_files.name
            self._lock = self.t_lock(file_or_files)
        self._lock._obtain_lock()