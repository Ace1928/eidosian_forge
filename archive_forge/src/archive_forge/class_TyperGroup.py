import errno
import inspect
import os
import sys
from enum import Enum
from gettext import gettext as _
from typing import (
import click
import click.core
import click.formatting
import click.parser
import click.shell_completion
import click.types
import click.utils
class TyperGroup(click.core.Group):

    def __init__(self, *, name: Optional[str]=None, commands: Optional[Union[Dict[str, click.Command], Sequence[click.Command]]]=None, rich_markup_mode: MarkupMode=None, rich_help_panel: Union[str, None]=None, **attrs: Any) -> None:
        super().__init__(name=name, commands=commands, **attrs)
        self.rich_markup_mode: MarkupMode = rich_markup_mode
        self.rich_help_panel = rich_help_panel

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        _typer_format_options(self, ctx=ctx, formatter=formatter)
        self.format_commands(ctx, formatter)

    def _main_shell_completion(self, ctx_args: MutableMapping[str, Any], prog_name: str, complete_var: Optional[str]=None) -> None:
        _typer_main_shell_completion(self, ctx_args=ctx_args, prog_name=prog_name, complete_var=complete_var)

    def main(self, args: Optional[Sequence[str]]=None, prog_name: Optional[str]=None, complete_var: Optional[str]=None, standalone_mode: bool=True, windows_expand_args: bool=True, **extra: Any) -> Any:
        return _main(self, args=args, prog_name=prog_name, complete_var=complete_var, standalone_mode=standalone_mode, windows_expand_args=windows_expand_args, **extra)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if not rich:
            return super().format_help(ctx, formatter)
        return rich_utils.rich_format_help(obj=self, ctx=ctx, markup_mode=self.rich_markup_mode)