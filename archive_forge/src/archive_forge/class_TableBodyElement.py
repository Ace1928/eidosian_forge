from __future__ import annotations
import sys
from typing import ClassVar, Dict, Iterable, List, Optional, Type, Union
from markdown_it import MarkdownIt
from markdown_it.token import Token
from rich.table import Table
from . import box
from ._loop import loop_first
from ._stack import Stack
from .console import Console, ConsoleOptions, JustifyMethod, RenderResult
from .containers import Renderables
from .jupyter import JupyterMixin
from .panel import Panel
from .rule import Rule
from .segment import Segment
from .style import Style, StyleStack
from .syntax import Syntax
from .text import Text, TextType
class TableBodyElement(MarkdownElement):
    """MarkdownElement corresponding to `tbody_open` and `tbody_close`."""

    def __init__(self) -> None:
        self.rows: list[TableRowElement] = []

    def on_child_close(self, context: 'MarkdownContext', child: 'MarkdownElement') -> bool:
        assert isinstance(child, TableRowElement)
        self.rows.append(child)
        return False