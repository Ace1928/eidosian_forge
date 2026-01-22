from __future__ import annotations
import argparse
import contextlib
import dataclasses
import difflib
import itertools
import re as _re
import shlex
import shutil
import sys
from gettext import gettext as _
from typing import Any, Dict, Generator, Iterable, List, NoReturn, Optional, Set, Tuple
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from typing_extensions import override
from . import _arguments, _strings, conf
from ._parsers import ParserSpecification
def _format_action(self, action: argparse.Action):
    invocation = self.formatter._format_action_invocation(action)
    indent = self.formatter._current_indent
    help_position = min(self.formatter._action_max_length + 4, self.formatter._max_help_position)
    if self.formatter._fixed_help_position:
        help_position = 4
    item_parts: List[RenderableType] = []
    if action.option_strings == ['-h', '--help']:
        assert action.help is not None
        action.help = str_from_rich(Text.from_markup('[helptext]' + action.help + '[/helptext]'))
    if action.help is not None:
        assert isinstance(action.help, str)
        helptext = Text.from_ansi(action.help.replace('%%', '%')) if _strings.strip_ansi_sequences(action.help) != action.help else Text.from_markup(action.help.replace('%%', '%'))
    else:
        helptext = Text('')
    if action.help and len(_strings.strip_ansi_sequences(invocation)) + indent < help_position - 1 and (not self.formatter._fixed_help_position):
        table = Table(show_header=False, box=None, padding=0)
        table.add_column(width=help_position - indent)
        table.add_column()
        table.add_row(Text.from_ansi(invocation, style=THEME.invocation), helptext)
        item_parts.append(table)
    else:
        item_parts.append(Text.from_ansi(invocation + '\n', style=THEME.invocation))
        if action.help:
            item_parts.append(Padding(helptext, pad=(0, 0, 0, help_position - indent)))
    try:
        subaction: argparse.Action
        for subaction in action._get_subactions():
            self.formatter._indent()
            item_parts.append(Padding(Group(*self._format_action(subaction)), pad=(0, 0, 0, self.formatter._indent_increment)))
            self.formatter._dedent()
    except AttributeError:
        pass
    return item_parts