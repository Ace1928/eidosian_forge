import inspect
import sys
from collections import defaultdict
from gettext import gettext as _
from os import getenv
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Union
import click
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, RenderableType, group
from rich.emoji import Emoji
from rich.highlighter import RegexHighlighter
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
def _make_rich_rext(*, text: str, style: str='', markup_mode: MarkupMode) -> Union[Markdown, Text]:
    """Take a string, remove indentations, and return styled text.

    By default, return the text as a Rich Text with the request style.
    If `rich_markdown_enable` is `True`, also parse the text for Rich markup strings.
    If `rich_markup_enable` is `True`, parse as Markdown.

    Only one of `rich_markdown_enable` or `rich_markup_enable` can be True.
    If both are True, `rich_markdown_enable` takes precedence.
    """
    text = inspect.cleandoc(text)
    if markup_mode == MARKUP_MODE_MARKDOWN:
        text = Emoji.replace(text)
        return Markdown(text, style=style)
    if markup_mode == MARKUP_MODE_RICH:
        return highlighter(Text.from_markup(text, style=style))
    else:
        return highlighter(Text(text, style=style))