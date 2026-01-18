from __future__ import annotations
import collections
import datetime
import functools
import glob
import itertools
import os
import random
import socket
import sqlite3
import string
import sys
import textwrap
import threading
import zlib
from typing import (
from coverage.debug import NoDebugging, auto_repr
from coverage.exceptions import CoverageException, DataError
from coverage.files import PathAliases
from coverage.misc import file_be_gone, isolate_module
from coverage.numbits import numbits_to_nums, numbits_union, nums_to_numbits
from coverage.sqlitedb import SqliteDb
from coverage.types import AnyCallable, FilePath, TArc, TDebugCtl, TLineNo, TWarnFn
from coverage.version import __version__
def touch_files(self, filenames: Collection[str], plugin_name: str | None=None) -> None:
    """Ensure that `filenames` appear in the data, empty if needed.

        `plugin_name` is the name of the plugin responsible for these files.
        It is used to associate the right filereporter, etc.
        """
    if self._debug.should('dataop'):
        self._debug.write(f'Touching {filenames!r}')
    self._start_using()
    with self._connect():
        if not self._has_arcs and (not self._has_lines):
            raise DataError("Can't touch files in an empty CoverageData")
        for filename in filenames:
            self._file_id(filename, add=True)
            if plugin_name:
                self.add_file_tracers({filename: plugin_name})