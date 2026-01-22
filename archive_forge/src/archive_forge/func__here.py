from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def _here() -> tuple[str, int]:
    frame = inspect.currentframe()
    assert frame is not None
    assert frame.f_back is not None
    info = inspect.getframeinfo(frame.f_back)
    return (info.filename, info.lineno)