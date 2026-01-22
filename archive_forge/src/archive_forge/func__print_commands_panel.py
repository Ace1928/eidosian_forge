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
def _print_commands_panel(*, name: str, commands: List[click.Command], markup_mode: MarkupMode, console: Console, cmd_len: int) -> None:
    t_styles: Dict[str, Any] = {'show_lines': STYLE_COMMANDS_TABLE_SHOW_LINES, 'leading': STYLE_COMMANDS_TABLE_LEADING, 'box': STYLE_COMMANDS_TABLE_BOX, 'border_style': STYLE_COMMANDS_TABLE_BORDER_STYLE, 'row_styles': STYLE_COMMANDS_TABLE_ROW_STYLES, 'pad_edge': STYLE_COMMANDS_TABLE_PAD_EDGE, 'padding': STYLE_COMMANDS_TABLE_PADDING}
    box_style = getattr(box, t_styles.pop('box'), None)
    commands_table = Table(highlight=False, show_header=False, expand=True, box=box_style, **t_styles)
    commands_table.add_column(style='bold cyan', no_wrap=True, width=cmd_len)
    commands_table.add_column('Description', justify='left', no_wrap=False, ratio=10)
    rows: List[List[Union[RenderableType, None]]] = []
    deprecated_rows: List[Union[RenderableType, None]] = []
    for command in commands:
        helptext = command.short_help or command.help or ''
        command_name = command.name or ''
        if command.deprecated:
            command_name_text = Text(f'{command_name}', style=STYLE_DEPRECATED_COMMAND)
            deprecated_rows.append(Text(DEPRECATED_STRING, style=STYLE_DEPRECATED))
        else:
            command_name_text = Text(command_name)
            deprecated_rows.append(None)
        rows.append([command_name_text, _make_command_help(help_text=helptext, markup_mode=markup_mode)])
    rows_with_deprecated = rows
    if any(deprecated_rows):
        rows_with_deprecated = []
        for row, deprecated_text in zip(rows, deprecated_rows):
            rows_with_deprecated.append([*row, deprecated_text])
    for row in rows_with_deprecated:
        commands_table.add_row(*row)
    if commands_table.row_count:
        console.print(Panel(commands_table, border_style=STYLE_COMMANDS_PANEL_BORDER, title=name, title_align=ALIGN_COMMANDS_PANEL))