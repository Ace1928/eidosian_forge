import collections
import inspect
import socket
import sys
import tempfile
import unittest
from typing import List, Tuple
from itertools import islice
from pathlib import Path
from unittest import mock
from bpython import config, repl, cli, autocomplete
from bpython.line import LinePart
from bpython.test import (
class FakeRepl(repl.Repl):

    def __init__(self, conf=None):
        super().__init__(repl.Interpreter(), setup_config(conf))
        self._current_line = ''
        self._cursor_offset = 0

    def _get_current_line(self) -> str:
        return self._current_line

    def _set_current_line(self, val: str) -> None:
        self._current_line = val

    def _get_cursor_offset(self) -> int:
        return self._cursor_offset

    def _set_cursor_offset(self, val: int) -> None:
        self._cursor_offset = val

    def getstdout(self) -> str:
        raise NotImplementedError

    def reprint_line(self, lineno: int, tokens: List[Tuple[repl._TokenType, str]]) -> None:
        raise NotImplementedError

    def reevaluate(self):
        raise NotImplementedError