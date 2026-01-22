from __future__ import annotations
import sys
from contextlib import contextmanager
from contextvars import ContextVar
class WasmUnsupportedError(Exception):
    pass