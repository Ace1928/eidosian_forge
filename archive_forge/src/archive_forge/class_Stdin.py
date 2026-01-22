from __future__ import annotations
import sys
import os
from contextlib import contextmanager
from typing import ContextManager, Iterator, TextIO, cast
from ..utils import DummyContext
from .base import PipeInput
from .vt100 import Vt100Input
class Stdin:
    encoding = 'utf-8'

    def isatty(stdin) -> bool:
        return True

    def fileno(stdin) -> int:
        return self.pipe.read_fd