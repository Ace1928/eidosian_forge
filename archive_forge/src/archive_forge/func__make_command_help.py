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
def _make_command_help(*, help_text: str, markup_mode: MarkupMode) -> Union[Text, Markdown]:
    """Build cli help text for a click group command.

    That is, when calling help on groups with multiple subcommands
    (not the main help text when calling the subcommand help).

    Returns the first paragraph of help text for a command, rendered either as a
    Rich Text object or as Markdown.
    Ignores single newlines as paragraph markers, looks for double only.
    """
    paragraphs = inspect.cleandoc(help_text).split('\n\n')
    if markup_mode != MARKUP_MODE_RICH and (not paragraphs[0].startswith('\x08')):
        paragraphs[0] = paragraphs[0].replace('\n', ' ')
    elif paragraphs[0].startswith('\x08'):
        paragraphs[0] = paragraphs[0].replace('\x08\n', '')
    return _make_rich_rext(text=paragraphs[0].strip(), style=STYLE_OPTION_HELP, markup_mode=markup_mode)