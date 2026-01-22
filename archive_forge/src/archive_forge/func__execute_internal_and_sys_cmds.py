import click
import os
import shlex
import sys
from collections import defaultdict
from .exceptions import CommandLineParserError, ExitReplException
def _execute_internal_and_sys_cmds(command, allow_internal_commands=True, allow_system_commands=True):
    """
    Executes internal, system, and all the other registered click commands from the input
    """
    if allow_system_commands and dispatch_repl_commands(command):
        return None
    if allow_internal_commands:
        result = handle_internal_commands(command)
        if isinstance(result, str):
            click.echo(result)
            return None
    try:
        return split_arg_string(command)
    except ValueError as e:
        raise CommandLineParserError('{}'.format(e))