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
def _file_id(self, filename: str, add: bool=False) -> int | None:
    """Get the file id for `filename`.

        If filename is not in the database yet, add it if `add` is True.
        If `add` is not True, return None.
        """
    if filename not in self._file_map:
        if add:
            with self._connect() as con:
                self._file_map[filename] = con.execute_for_rowid('insert or replace into file (path) values (?)', (filename,))
    return self._file_map.get(filename)