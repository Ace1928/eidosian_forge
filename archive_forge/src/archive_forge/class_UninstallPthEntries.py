import functools
import os
import sys
import sysconfig
from importlib.util import cache_from_source
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple
from pip._internal.exceptions import UninstallationError
from pip._internal.locations import get_bin_prefix, get_bin_user
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.logging import getLogger, indent_log
from pip._internal.utils.misc import ask, normalize_path, renames, rmtree
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory
from pip._internal.utils.virtualenv import running_under_virtualenv
class UninstallPthEntries:

    def __init__(self, pth_file: str) -> None:
        self.file = pth_file
        self.entries: Set[str] = set()
        self._saved_lines: Optional[List[bytes]] = None

    def add(self, entry: str) -> None:
        entry = os.path.normcase(entry)
        if WINDOWS and (not os.path.splitdrive(entry)[0]):
            entry = entry.replace('\\', '/')
        self.entries.add(entry)

    def remove(self) -> None:
        logger.verbose('Removing pth entries from %s:', self.file)
        if not os.path.isfile(self.file):
            logger.warning('Cannot remove entries from nonexistent file %s', self.file)
            return
        with open(self.file, 'rb') as fh:
            lines = fh.readlines()
            self._saved_lines = lines
        if any((b'\r\n' in line for line in lines)):
            endline = '\r\n'
        else:
            endline = '\n'
        if lines and (not lines[-1].endswith(endline.encode('utf-8'))):
            lines[-1] = lines[-1] + endline.encode('utf-8')
        for entry in self.entries:
            try:
                logger.verbose('Removing entry: %s', entry)
                lines.remove((entry + endline).encode('utf-8'))
            except ValueError:
                pass
        with open(self.file, 'wb') as fh:
            fh.writelines(lines)

    def rollback(self) -> bool:
        if self._saved_lines is None:
            logger.error('Cannot roll back changes to %s, none were made', self.file)
            return False
        logger.debug('Rolling %s back to previous state', self.file)
        with open(self.file, 'wb') as fh:
            fh.writelines(self._saved_lines)
        return True