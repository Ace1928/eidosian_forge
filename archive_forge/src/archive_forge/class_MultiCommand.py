import enum
import errno
import inspect
import os
import sys
import typing as t
from collections import abc
from contextlib import contextmanager
from contextlib import ExitStack
from functools import update_wrapper
from gettext import gettext as _
from gettext import ngettext
from itertools import repeat
from types import TracebackType
from . import types
from .exceptions import Abort
from .exceptions import BadParameter
from .exceptions import ClickException
from .exceptions import Exit
from .exceptions import MissingParameter
from .exceptions import UsageError
from .formatting import HelpFormatter
from .formatting import join_options
from .globals import pop_context
from .globals import push_context
from .parser import _flag_needs_value
from .parser import OptionParser
from .parser import split_opt
from .termui import confirm
from .termui import prompt
from .termui import style
from .utils import _detect_program_name
from .utils import _expand_args
from .utils import echo
from .utils import make_default_short_help
from .utils import make_str
from .utils import PacifyFlushWrapper
class MultiCommand(Command):
    """A multi command is the basic implementation of a command that
    dispatches to subcommands.  The most common version is the
    :class:`Group`.

    :param invoke_without_command: this controls how the multi command itself
                                   is invoked.  By default it's only invoked
                                   if a subcommand is provided.
    :param no_args_is_help: this controls what happens if no arguments are
                            provided.  This option is enabled by default if
                            `invoke_without_command` is disabled or disabled
                            if it's enabled.  If enabled this will add
                            ``--help`` as argument if no arguments are
                            passed.
    :param subcommand_metavar: the string that is used in the documentation
                               to indicate the subcommand place.
    :param chain: if this is set to `True` chaining of multiple subcommands
                  is enabled.  This restricts the form of commands in that
                  they cannot have optional arguments but it allows
                  multiple commands to be chained together.
    :param result_callback: The result callback to attach to this multi
        command. This can be set or changed later with the
        :meth:`result_callback` decorator.
    :param attrs: Other command arguments described in :class:`Command`.
    """
    allow_extra_args = True
    allow_interspersed_args = False

    def __init__(self, name: t.Optional[str]=None, invoke_without_command: bool=False, no_args_is_help: t.Optional[bool]=None, subcommand_metavar: t.Optional[str]=None, chain: bool=False, result_callback: t.Optional[t.Callable[..., t.Any]]=None, **attrs: t.Any) -> None:
        super().__init__(name, **attrs)
        if no_args_is_help is None:
            no_args_is_help = not invoke_without_command
        self.no_args_is_help = no_args_is_help
        self.invoke_without_command = invoke_without_command
        if subcommand_metavar is None:
            if chain:
                subcommand_metavar = 'COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...'
            else:
                subcommand_metavar = 'COMMAND [ARGS]...'
        self.subcommand_metavar = subcommand_metavar
        self.chain = chain
        self._result_callback = result_callback
        if self.chain:
            for param in self.params:
                if isinstance(param, Argument) and (not param.required):
                    raise RuntimeError('Multi commands in chain mode cannot have optional arguments.')

    def to_info_dict(self, ctx: Context) -> t.Dict[str, t.Any]:
        info_dict = super().to_info_dict(ctx)
        commands = {}
        for name in self.list_commands(ctx):
            command = self.get_command(ctx, name)
            if command is None:
                continue
            sub_ctx = ctx._make_sub_context(command)
            with sub_ctx.scope(cleanup=False):
                commands[name] = command.to_info_dict(sub_ctx)
        info_dict.update(commands=commands, chain=self.chain)
        return info_dict

    def collect_usage_pieces(self, ctx: Context) -> t.List[str]:
        rv = super().collect_usage_pieces(ctx)
        rv.append(self.subcommand_metavar)
        return rv

    def format_options(self, ctx: Context, formatter: HelpFormatter) -> None:
        super().format_options(ctx, formatter)
        self.format_commands(ctx, formatter)

    def result_callback(self, replace: bool=False) -> t.Callable[[F], F]:
        """Adds a result callback to the command.  By default if a
        result callback is already registered this will chain them but
        this can be disabled with the `replace` parameter.  The result
        callback is invoked with the return value of the subcommand
        (or the list of return values from all subcommands if chaining
        is enabled) as well as the parameters as they would be passed
        to the main callback.

        Example::

            @click.group()
            @click.option('-i', '--input', default=23)
            def cli(input):
                return 42

            @cli.result_callback()
            def process_result(result, input):
                return result + input

        :param replace: if set to `True` an already existing result
                        callback will be removed.

        .. versionchanged:: 8.0
            Renamed from ``resultcallback``.

        .. versionadded:: 3.0
        """

        def decorator(f: F) -> F:
            old_callback = self._result_callback
            if old_callback is None or replace:
                self._result_callback = f
                return f

            def function(__value, *args, **kwargs):
                inner = old_callback(__value, *args, **kwargs)
                return f(inner, *args, **kwargs)
            self._result_callback = rv = update_wrapper(t.cast(F, function), f)
            return rv
        return decorator

    def format_commands(self, ctx: Context, formatter: HelpFormatter) -> None:
        """Extra format methods for multi methods that adds all the commands
        after the options.
        """
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None:
                continue
            if cmd.hidden:
                continue
            commands.append((subcommand, cmd))
        if len(commands):
            limit = formatter.width - 6 - max((len(cmd[0]) for cmd in commands))
            rows = []
            for subcommand, cmd in commands:
                help = cmd.get_short_help_str(limit)
                rows.append((subcommand, help))
            if rows:
                with formatter.section(_('Commands')):
                    formatter.write_dl(rows)

    def parse_args(self, ctx: Context, args: t.List[str]) -> t.List[str]:
        if not args and self.no_args_is_help and (not ctx.resilient_parsing):
            echo(ctx.get_help(), color=ctx.color)
            ctx.exit()
        rest = super().parse_args(ctx, args)
        if self.chain:
            ctx.protected_args = rest
            ctx.args = []
        elif rest:
            ctx.protected_args, ctx.args = (rest[:1], rest[1:])
        return ctx.args

    def invoke(self, ctx: Context) -> t.Any:

        def _process_result(value: t.Any) -> t.Any:
            if self._result_callback is not None:
                value = ctx.invoke(self._result_callback, value, **ctx.params)
            return value
        if not ctx.protected_args:
            if self.invoke_without_command:
                with ctx:
                    rv = super().invoke(ctx)
                    return _process_result([] if self.chain else rv)
            ctx.fail(_('Missing command.'))
        args = [*ctx.protected_args, *ctx.args]
        ctx.args = []
        ctx.protected_args = []
        if not self.chain:
            with ctx:
                cmd_name, cmd, args = self.resolve_command(ctx, args)
                assert cmd is not None
                ctx.invoked_subcommand = cmd_name
                super().invoke(ctx)
                sub_ctx = cmd.make_context(cmd_name, args, parent=ctx)
                with sub_ctx:
                    return _process_result(sub_ctx.command.invoke(sub_ctx))
        with ctx:
            ctx.invoked_subcommand = '*' if args else None
            super().invoke(ctx)
            contexts = []
            while args:
                cmd_name, cmd, args = self.resolve_command(ctx, args)
                assert cmd is not None
                sub_ctx = cmd.make_context(cmd_name, args, parent=ctx, allow_extra_args=True, allow_interspersed_args=False)
                contexts.append(sub_ctx)
                args, sub_ctx.args = (sub_ctx.args, [])
            rv = []
            for sub_ctx in contexts:
                with sub_ctx:
                    rv.append(sub_ctx.command.invoke(sub_ctx))
            return _process_result(rv)

    def resolve_command(self, ctx: Context, args: t.List[str]) -> t.Tuple[t.Optional[str], t.Optional[Command], t.List[str]]:
        cmd_name = make_str(args[0])
        original_cmd_name = cmd_name
        cmd = self.get_command(ctx, cmd_name)
        if cmd is None and ctx.token_normalize_func is not None:
            cmd_name = ctx.token_normalize_func(cmd_name)
            cmd = self.get_command(ctx, cmd_name)
        if cmd is None and (not ctx.resilient_parsing):
            if split_opt(cmd_name)[0]:
                self.parse_args(ctx, ctx.args)
            ctx.fail(_('No such command {name!r}.').format(name=original_cmd_name))
        return (cmd_name if cmd else None, cmd, args[1:])

    def get_command(self, ctx: Context, cmd_name: str) -> t.Optional[Command]:
        """Given a context and a command name, this returns a
        :class:`Command` object if it exists or returns `None`.
        """
        raise NotImplementedError

    def list_commands(self, ctx: Context) -> t.List[str]:
        """Returns a list of subcommand names in the order they should
        appear.
        """
        return []

    def shell_complete(self, ctx: Context, incomplete: str) -> t.List['CompletionItem']:
        """Return a list of completions for the incomplete value. Looks
        at the names of options, subcommands, and chained
        multi-commands.

        :param ctx: Invocation context for this command.
        :param incomplete: Value being completed. May be empty.

        .. versionadded:: 8.0
        """
        from click.shell_completion import CompletionItem
        results = [CompletionItem(name, help=command.get_short_help_str()) for name, command in _complete_visible_commands(ctx, incomplete)]
        results.extend(super().shell_complete(ctx, incomplete))
        return results