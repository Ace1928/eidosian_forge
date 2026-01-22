import os
import re
import typing as t
from gettext import gettext as _
from .core import Argument
from .core import BaseCommand
from .core import Context
from .core import MultiCommand
from .core import Option
from .core import Parameter
from .core import ParameterSource
from .parser import split_arg_string
from .utils import echo
def _resolve_context(cli: BaseCommand, ctx_args: t.MutableMapping[str, t.Any], prog_name: str, args: t.List[str]) -> Context:
    """Produce the context hierarchy starting with the command and
    traversing the complete arguments. This only follows the commands,
    it doesn't trigger input prompts or callbacks.

    :param cli: Command being called.
    :param prog_name: Name of the executable in the shell.
    :param args: List of complete args before the incomplete value.
    """
    ctx_args['resilient_parsing'] = True
    ctx = cli.make_context(prog_name, args.copy(), **ctx_args)
    args = ctx.protected_args + ctx.args
    while args:
        command = ctx.command
        if isinstance(command, MultiCommand):
            if not command.chain:
                name, cmd, args = command.resolve_command(ctx, args)
                if cmd is None:
                    return ctx
                ctx = cmd.make_context(name, args, parent=ctx, resilient_parsing=True)
                args = ctx.protected_args + ctx.args
            else:
                sub_ctx = ctx
                while args:
                    name, cmd, args = command.resolve_command(ctx, args)
                    if cmd is None:
                        return ctx
                    sub_ctx = cmd.make_context(name, args, parent=ctx, allow_extra_args=True, allow_interspersed_args=False, resilient_parsing=True)
                    args = sub_ctx.args
                ctx = sub_ctx
                args = [*sub_ctx.protected_args, *sub_ctx.args]
        else:
            break
    return ctx