from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
class DebugOutputFile:
    """A file-like object that includes pid and cwd information."""

    def __init__(self, outfile: IO[str] | None, filters: Iterable[Callable[[str], str]]):
        self.outfile = outfile
        self.filters = list(filters)
        self.pid = os.getpid()

    @classmethod
    def get_one(cls, fileobj: IO[str] | None=None, file_name: str | None=None, filters: Iterable[Callable[[str], str]]=(), interim: bool=False) -> DebugOutputFile:
        """Get a DebugOutputFile.

        If `fileobj` is provided, then a new DebugOutputFile is made with it.

        If `fileobj` isn't provided, then a file is chosen (`file_name` if
        provided, or COVERAGE_DEBUG_FILE, or stderr), and a process-wide
        singleton DebugOutputFile is made.

        `filters` are the text filters to apply to the stream to annotate with
        pids, etc.

        If `interim` is true, then a future `get_one` can replace this one.

        """
        if fileobj is not None:
            return cls(fileobj, filters)
        the_one, is_interim = cls._get_singleton_data()
        if the_one is None or is_interim:
            if file_name is not None:
                fileobj = open(file_name, 'a', encoding='utf-8')
            else:
                file_name = os.getenv('COVERAGE_DEBUG_FILE', FORCED_DEBUG_FILE)
                if file_name in ('stdout', 'stderr'):
                    fileobj = getattr(sys, file_name)
                elif file_name:
                    fileobj = open(file_name, 'a', encoding='utf-8')
                    atexit.register(fileobj.close)
                else:
                    fileobj = sys.stderr
            the_one = cls(fileobj, filters)
            cls._set_singleton_data(the_one, interim)
        if not the_one.filters:
            the_one.filters = list(filters)
        return the_one
    SYS_MOD_NAME = '$coverage.debug.DebugOutputFile.the_one'
    SINGLETON_ATTR = 'the_one_and_is_interim'

    @classmethod
    def _set_singleton_data(cls, the_one: DebugOutputFile, interim: bool) -> None:
        """Set the one DebugOutputFile to rule them all."""
        singleton_module = types.ModuleType(cls.SYS_MOD_NAME)
        setattr(singleton_module, cls.SINGLETON_ATTR, (the_one, interim))
        sys.modules[cls.SYS_MOD_NAME] = singleton_module

    @classmethod
    def _get_singleton_data(cls) -> tuple[DebugOutputFile | None, bool]:
        """Get the one DebugOutputFile."""
        singleton_module = sys.modules.get(cls.SYS_MOD_NAME)
        return getattr(singleton_module, cls.SINGLETON_ATTR, (None, True))

    @classmethod
    def _del_singleton_data(cls) -> None:
        """Delete the one DebugOutputFile, just for tests to use."""
        if cls.SYS_MOD_NAME in sys.modules:
            del sys.modules[cls.SYS_MOD_NAME]

    def write(self, text: str) -> None:
        """Just like file.write, but filter through all our filters."""
        assert self.outfile is not None
        self.outfile.write(filter_text(text, self.filters))
        self.outfile.flush()

    def flush(self) -> None:
        """Flush our file."""
        assert self.outfile is not None
        self.outfile.flush()