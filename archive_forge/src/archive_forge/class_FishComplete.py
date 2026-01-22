import os
import re
import sys
from typing import Any, Dict, List, Tuple
import click
import click.parser
import click.shell_completion
from ._completion_shared import (
class FishComplete(click.shell_completion.FishComplete):
    name = Shells.fish.value
    source_template = COMPLETION_SCRIPT_FISH

    def source_vars(self) -> Dict[str, Any]:
        return {'complete_func': self.func_name, 'autocomplete_var': self.complete_var, 'prog_name': self.prog_name}

    def get_completion_args(self) -> Tuple[List[str], str]:
        completion_args = os.getenv('_TYPER_COMPLETE_ARGS', '')
        cwords = click.parser.split_arg_string(completion_args)
        args = cwords[1:]
        if args and (not completion_args.endswith(' ')):
            incomplete = args[-1]
            args = args[:-1]
        else:
            incomplete = ''
        return (args, incomplete)

    def format_completion(self, item: click.shell_completion.CompletionItem) -> str:
        if item.help:
            formatted_help = re.sub('\\s', ' ', item.help)
            return f'{item.value}\t{formatted_help}'
        else:
            return f'{item.value}'

    def complete(self) -> str:
        complete_action = os.getenv('_TYPER_COMPLETE_FISH_ACTION', '')
        args, incomplete = self.get_completion_args()
        completions = self.get_completions(args, incomplete)
        show_args = [self.format_completion(item) for item in completions]
        if complete_action == 'get-args':
            if show_args:
                return '\n'.join(show_args)
        elif complete_action == 'is-args':
            if show_args:
                sys.exit(0)
            else:
                sys.exit(1)
        return ''