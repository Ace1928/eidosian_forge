from __future__ import annotations
import contextvars
import datetime
import functools
import os
import signal
import threading
import traceback
from typing import (
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.filters import Condition, is_done, renderer_height_is_known
from prompt_toolkit.formatted_text import (
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import (
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.layout.dimension import AnyDimension, D
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.utils import in_main_thread
from .formatters import Formatter, create_default_formatters
class _ProgressControl(UIControl):
    """
    User control for the progress bar.
    """

    def __init__(self, progress_bar: ProgressBar, formatter: Formatter, cancel_callback: Callable[[], None] | None) -> None:
        self.progress_bar = progress_bar
        self.formatter = formatter
        self._key_bindings = create_key_bindings(cancel_callback)

    def create_content(self, width: int, height: int) -> UIContent:
        items: list[StyleAndTextTuples] = []
        for pr in self.progress_bar.counters:
            try:
                text = self.formatter.format(self.progress_bar, pr, width)
            except BaseException:
                traceback.print_exc()
                text = 'ERROR'
            items.append(to_formatted_text(text))

        def get_line(i: int) -> StyleAndTextTuples:
            return items[i]
        return UIContent(get_line=get_line, line_count=len(items), show_cursor=False)

    def is_focusable(self) -> bool:
        return True

    def get_key_bindings(self) -> KeyBindings:
        return self._key_bindings