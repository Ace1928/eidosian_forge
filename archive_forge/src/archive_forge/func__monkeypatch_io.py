from __future__ import annotations
import logging # isort:skip
from contextlib import contextmanager
from os.path import basename, splitext
from types import ModuleType
from typing import Any, Callable, ClassVar
from ...core.types import PathLike
from ...document import Document
from ...io.doc import curdoc, patch_curdoc
from .code_runner import CodeRunner
from .handler import Handler
@contextmanager
def _monkeypatch_io(loggers: dict[str, Callable[..., None]]) -> dict[str, Any]:
    import bokeh.io as io
    old: dict[str, Any] = {}
    for f in CodeHandler._io_functions:
        old[f] = getattr(io, f)
        setattr(io, f, loggers[f])
    yield
    for f in old:
        setattr(io, f, old[f])