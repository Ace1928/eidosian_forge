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
def _main_shell_completion(self, ctx_args: MutableMapping[str, Any], prog_name: str, complete_var: Optional[str]=None) -> None:
    _typer_main_shell_completion(self, ctx_args=ctx_args, prog_name=prog_name, complete_var=complete_var)