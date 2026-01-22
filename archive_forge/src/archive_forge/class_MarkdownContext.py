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
class MarkdownContext:
    """Manages the console render state."""

    def __init__(self, console: Console, options: ConsoleOptions, style: Style, inline_code_lexer: Optional[str]=None, inline_code_theme: str='monokai') -> None:
        self.console = console
        self.options = options
        self.style_stack: StyleStack = StyleStack(style)
        self.stack: Stack[MarkdownElement] = Stack()
        self._syntax: Optional[Syntax] = None
        if inline_code_lexer is not None:
            self._syntax = Syntax('', inline_code_lexer, theme=inline_code_theme)

    @property
    def current_style(self) -> Style:
        """Current style which is the product of all styles on the stack."""
        return self.style_stack.current

    def on_text(self, text: str, node_type: str) -> None:
        """Called when the parser visits text."""
        if node_type in {'fence', 'code_inline'} and self._syntax is not None:
            highlight_text = self._syntax.highlight(text)
            highlight_text.rstrip()
            self.stack.top.on_text(self, Text.assemble(highlight_text, style=self.style_stack.current))
        else:
            self.stack.top.on_text(self, text)

    def enter_style(self, style_name: Union[str, Style]) -> Style:
        """Enter a style context."""
        style = self.console.get_style(style_name, default='none')
        self.style_stack.push(style)
        return self.current_style

    def leave_style(self) -> Style:
        """Leave a style context."""
        style = self.style_stack.pop()
        return style