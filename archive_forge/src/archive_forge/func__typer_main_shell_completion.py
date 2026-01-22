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
def _typer_main_shell_completion(self: click.core.Command, *, ctx_args: MutableMapping[str, Any], prog_name: str, complete_var: Optional[str]=None) -> None:
    if complete_var is None:
        complete_var = f'_{prog_name}_COMPLETE'.replace('-', '_').upper()
    instruction = os.environ.get(complete_var)
    if not instruction:
        return
    from .completion import shell_complete
    rv = shell_complete(self, ctx_args, prog_name, complete_var, instruction)
    sys.exit(rv)