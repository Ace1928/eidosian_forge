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
class TableDataElement(MarkdownElement):
    """MarkdownElement corresponding to `td_open` and `td_close`
    and `th_open` and `th_close`."""

    @classmethod
    def create(cls, markdown: 'Markdown', token: Token) -> 'MarkdownElement':
        style = str(token.attrs.get('style')) or ''
        justify: JustifyMethod
        if 'text-align:right' in style:
            justify = 'right'
        elif 'text-align:center' in style:
            justify = 'center'
        elif 'text-align:left' in style:
            justify = 'left'
        else:
            justify = 'default'
        assert justify in get_args(JustifyMethod)
        return cls(justify=justify)

    def __init__(self, justify: JustifyMethod) -> None:
        self.content: Text = Text('', justify=justify)
        self.justify = justify

    def on_text(self, context: 'MarkdownContext', text: TextType) -> None:
        text = Text(text) if isinstance(text, str) else text
        text.stylize(context.current_style)
        self.content.append_text(text)