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
def _get_rich_console(stderr: bool=False) -> Console:
    return Console(theme=Theme({'option': STYLE_OPTION, 'switch': STYLE_SWITCH, 'negative_option': STYLE_NEGATIVE_OPTION, 'negative_switch': STYLE_NEGATIVE_SWITCH, 'metavar': STYLE_METAVAR, 'metavar_sep': STYLE_METAVAR_SEPARATOR, 'usage': STYLE_USAGE}), highlighter=highlighter, color_system=COLOR_SYSTEM, force_terminal=FORCE_TERMINAL, width=MAX_WIDTH, stderr=stderr)