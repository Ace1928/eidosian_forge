from __future__ import unicode_literals
import os
from glob import iglob
import click
from prompt_toolkit.completion import Completion, Completer
from .utils import _resolve_context, split_arg_string
def _get_completion_for_cmd_args(self, ctx_command, incomplete, autocomplete_ctx, args):
    choices = []
    param_called = False
    for param in ctx_command.params:
        if isinstance(param.type, click.types.UnprocessedParamType):
            return []
        elif getattr(param, 'hidden', False):
            continue
        elif isinstance(param, click.Option):
            for option in param.opts + param.secondary_opts:
                if option in args[param.nargs * -1:]:
                    param_called = True
                    break
                elif option.startswith(incomplete):
                    choices.append(Completion(text_type(option), -len(incomplete), display_meta=text_type(param.help or '')))
            if param_called:
                choices = self._get_completion_from_params(autocomplete_ctx, args, param, incomplete)
        elif isinstance(param, click.Argument):
            choices.extend(self._get_completion_from_params(autocomplete_ctx, args, param, incomplete))
    return choices