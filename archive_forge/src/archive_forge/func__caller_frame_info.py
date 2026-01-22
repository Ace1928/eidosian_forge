import inspect
import os
import platform
import sys
import threading
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from getpass import getpass
from html import escape
from inspect import isclass
from itertools import islice
from math import ceil
from time import monotonic
from types import FrameType, ModuleType, TracebackType
from typing import (
from pip._vendor.rich._null_file import NULL_FILE
from . import errors, themes
from ._emoji_replace import _emoji_replace
from ._export_format import CONSOLE_HTML_FORMAT, CONSOLE_SVG_FORMAT
from ._fileno import get_fileno
from ._log_render import FormatTimeCallable, LogRender
from .align import Align, AlignMethod
from .color import ColorSystem, blend_rgb
from .control import Control
from .emoji import EmojiVariant
from .highlighter import NullHighlighter, ReprHighlighter
from .markup import render as render_markup
from .measure import Measurement, measure_renderables
from .pager import Pager, SystemPager
from .pretty import Pretty, is_expandable
from .protocol import rich_cast
from .region import Region
from .scope import render_scope
from .screen import Screen
from .segment import Segment
from .style import Style, StyleType
from .styled import Styled
from .terminal_theme import DEFAULT_TERMINAL_THEME, SVG_EXPORT_THEME, TerminalTheme
from .text import Text, TextType
from .theme import Theme, ThemeStack
@staticmethod
def _caller_frame_info(offset: int, currentframe: Callable[[], Optional[FrameType]]=inspect.currentframe) -> Tuple[str, int, Dict[str, Any]]:
    """Get caller frame information.

        Args:
            offset (int): the caller offset within the current frame stack.
            currentframe (Callable[[], Optional[FrameType]], optional): the callable to use to
                retrieve the current frame. Defaults to ``inspect.currentframe``.

        Returns:
            Tuple[str, int, Dict[str, Any]]: A tuple containing the filename, the line number and
                the dictionary of local variables associated with the caller frame.

        Raises:
            RuntimeError: If the stack offset is invalid.
        """
    offset += 1
    frame = currentframe()
    if frame is not None:
        while offset and frame is not None:
            frame = frame.f_back
            offset -= 1
        assert frame is not None
        return (frame.f_code.co_filename, frame.f_lineno, frame.f_locals)
    else:
        frame_info = inspect.stack()[offset]
        return (frame_info.filename, frame_info.lineno, frame_info.frame.f_locals)