import sys
from threading import Event, RLock, Thread
from types import TracebackType
from typing import IO, Any, Callable, List, Optional, TextIO, Type, cast
from . import get_console
from .console import Console, ConsoleRenderable, RenderableType, RenderHook
from .control import Control
from .file_proxy import FileProxy
from .jupyter import JupyterMixin
from .live_render import LiveRender, VerticalOverflowMethod
from .screen import Screen
from .text import Text
def _disable_redirect_io(self) -> None:
    """Disable redirecting of stdout / stderr."""
    if self._restore_stdout:
        sys.stdout = cast('TextIO', self._restore_stdout)
        self._restore_stdout = None
    if self._restore_stderr:
        sys.stderr = cast('TextIO', self._restore_stderr)
        self._restore_stderr = None