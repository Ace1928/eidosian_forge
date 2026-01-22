import os
import re
import sys
from typing import Any, Dict, List, Tuple
import click
import click.parser
import click.shell_completion
from ._completion_shared import (
class BashComplete(click.shell_completion.BashComplete):
    name = Shells.bash.value
    source_template = COMPLETION_SCRIPT_BASH

    def source_vars(self) -> Dict[str, Any]:
        return {'complete_func': self.func_name, 'autocomplete_var': self.complete_var, 'prog_name': self.prog_name}

    def get_completion_args(self) -> Tuple[List[str], str]:
        cwords = click.parser.split_arg_string(os.environ['COMP_WORDS'])
        cword = int(os.environ['COMP_CWORD'])
        args = cwords[1:cword]
        try:
            incomplete = cwords[cword]
        except IndexError:
            incomplete = ''
        return (args, incomplete)

    def format_completion(self, item: click.shell_completion.CompletionItem) -> str:
        return f'{item.value}'

    def complete(self) -> str:
        args, incomplete = self.get_completion_args()
        completions = self.get_completions(args, incomplete)
        out = [self.format_completion(item) for item in completions]
        return '\n'.join(out)