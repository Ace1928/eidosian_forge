from __future__ import annotations  # isort: split
import __future__  # Regular import, not special!
import enum
import functools
import importlib
import inspect
import json
import socket as stdlib_socket
import sys
import types
from pathlib import Path, PurePath
from types import ModuleType
from typing import TYPE_CHECKING, Protocol
import attrs
import pytest
import trio
import trio.testing
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from .. import _core, _util
from .._core._tests.tutil import slow
from .pytest_plugin import RUN_SLOW
def _ensure_mypy_cache_updated() -> None:
    try:
        from mypy.api import run
    except ImportError as error:
        skip_if_optional_else_raise(error)
    global mypy_cache_updated
    if not mypy_cache_updated:
        result = run(['--config-file=', '--cache-dir=./.mypy_cache', '--no-error-summary', '-c', 'import trio'])
        assert not result[1]
        assert not result[0]
        mypy_cache_updated = True