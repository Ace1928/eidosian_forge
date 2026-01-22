import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _redirect_complete(self, text: str, line: str, begidx: int, endidx: int, compfunc: CompleterFunc) -> List[str]:
    """Called by complete() as the first tab completion function for all commands
        It determines if it should tab complete for redirection (|, >, >>) or use the
        completer function for the current command

        :param text: the string prefix we are attempting to match (all matches must begin with it)
        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :param compfunc: the completer function for the current command
                         this will be called if we aren't completing for redirection
        :return: a list of possible tab completions
        """
    _, raw_tokens = self.tokens_for_completion(line, begidx, endidx)
    if not raw_tokens:
        return []
    if len(raw_tokens) > 1:
        has_redirection = False
        in_pipe = False
        in_file_redir = False
        do_shell_completion = False
        do_path_completion = False
        prior_token = None
        for cur_token in raw_tokens:
            if cur_token in constants.REDIRECTION_TOKENS:
                has_redirection = True
                if cur_token == constants.REDIRECTION_PIPE:
                    if prior_token == constants.REDIRECTION_PIPE:
                        return []
                    in_pipe = True
                    in_file_redir = False
                else:
                    if prior_token in constants.REDIRECTION_TOKENS or in_file_redir:
                        return []
                    in_pipe = False
                    in_file_redir = True
            elif self.allow_redirection:
                do_shell_completion = False
                do_path_completion = False
                if prior_token == constants.REDIRECTION_PIPE:
                    do_shell_completion = True
                elif in_pipe or prior_token in (constants.REDIRECTION_OUTPUT, constants.REDIRECTION_APPEND):
                    do_path_completion = True
            prior_token = cur_token
        if do_shell_completion:
            return self.shell_cmd_complete(text, line, begidx, endidx)
        elif do_path_completion:
            return self.path_complete(text, line, begidx, endidx)
        elif has_redirection:
            return []
    return compfunc(text, line, begidx, endidx)