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
def _choose_lines_or_arcs(self, lines: bool=False, arcs: bool=False) -> None:
    """Force the data file to choose between lines and arcs."""
    assert lines or arcs
    assert not (lines and arcs)
    if lines and self._has_arcs:
        if self._debug.should('dataop'):
            self._debug.write("Error: Can't add line measurements to existing branch data")
        raise DataError("Can't add line measurements to existing branch data")
    if arcs and self._has_lines:
        if self._debug.should('dataop'):
            self._debug.write("Error: Can't add branch measurements to existing line data")
        raise DataError("Can't add branch measurements to existing line data")
    if not self._has_arcs and (not self._has_lines):
        self._has_lines = lines
        self._has_arcs = arcs
        with self._connect() as con:
            con.execute_void('insert or ignore into meta (key, value) values (?, ?)', ('has_arcs', str(int(arcs))))