from __future__ import annotations
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import Any, Dict, Tuple, cast
import pytest
import sniffio
from ._core._eventloop import get_all_backends, get_async_backend
from .abc import TestRunner
@pytest.fixture(scope='module', params=get_all_backends())
def anyio_backend(request: Any) -> Any:
    return request.param