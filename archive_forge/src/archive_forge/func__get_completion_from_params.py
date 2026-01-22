from __future__ import unicode_literals
import os
from glob import iglob
import click
from prompt_toolkit.completion import Completion, Completer
from .utils import _resolve_context, split_arg_string
def _get_completion_from_params(self, autocomplete_ctx, args, param, incomplete):
    choices = []
    param_type = param.type
    if not HAS_CLICK_V8 and isinstance(param_type, click.Choice):
        choices.extend(self._get_completion_from_choices_click_le_7(param, incomplete))
    elif isinstance(param_type, click.types.BoolParamType):
        choices.extend(self._get_completion_for_Boolean_type(param, incomplete))
    elif isinstance(param_type, (click.Path, click.File)):
        choices.extend(self._get_completion_for_Path_types(param, args, incomplete))
    elif getattr(param, AUTO_COMPLETION_PARAM, None) is not None:
        choices.extend(self._get_completion_from_autocompletion_functions(param, autocomplete_ctx, args, incomplete))
    return choices