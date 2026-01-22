import io
import sys
import typing
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from io import RawIOBase, UnsupportedOperation
from math import ceil
from mmap import mmap
from operator import length_hint
from os import PathLike, stat
from threading import Event, RLock, Thread
from types import TracebackType
from typing import (
from . import filesize, get_console
from .console import Console, Group, JustifyMethod, RenderableType
from .highlighter import Highlighter
from .jupyter import JupyterMixin
from .live import Live
from .progress_bar import ProgressBar
from .spinner import Spinner
from .style import StyleType
from .table import Column, Table
from .text import Text, TextType
class SpinnerColumn(ProgressColumn):
    """A column with a 'spinner' animation.

    Args:
        spinner_name (str, optional): Name of spinner animation. Defaults to "dots".
        style (StyleType, optional): Style of spinner. Defaults to "progress.spinner".
        speed (float, optional): Speed factor of spinner. Defaults to 1.0.
        finished_text (TextType, optional): Text used when task is finished. Defaults to " ".
    """

    def __init__(self, spinner_name: str='dots', style: Optional[StyleType]='progress.spinner', speed: float=1.0, finished_text: TextType=' ', table_column: Optional[Column]=None):
        self.spinner = Spinner(spinner_name, style=style, speed=speed)
        self.finished_text = Text.from_markup(finished_text) if isinstance(finished_text, str) else finished_text
        super().__init__(table_column=table_column)

    def set_spinner(self, spinner_name: str, spinner_style: Optional[StyleType]='progress.spinner', speed: float=1.0) -> None:
        """Set a new spinner.

        Args:
            spinner_name (str): Spinner name, see python -m rich.spinner.
            spinner_style (Optional[StyleType], optional): Spinner style. Defaults to "progress.spinner".
            speed (float, optional): Speed factor of spinner. Defaults to 1.0.
        """
        self.spinner = Spinner(spinner_name, style=spinner_style, speed=speed)

    def render(self, task: 'Task') -> RenderableType:
        text = self.finished_text if task.finished else self.spinner.render(task.get_time())
        return text