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
def tokens_for_completion(self, line: str, begidx: int, endidx: int) -> Tuple[List[str], List[str]]:
    """Used by tab completion functions to get all tokens through the one being completed.

        :param line: the current input line with leading whitespace removed
        :param begidx: the beginning index of the prefix text
        :param endidx: the ending index of the prefix text
        :return: A 2 item tuple where the items are
                 **On Success**
                 - tokens: list of unquoted tokens - this is generally the list needed for tab completion functions
                 - raw_tokens: list of tokens with any quotes preserved = this can be used to know if a token was quoted
                 or is missing a closing quote
                 Both lists are guaranteed to have at least 1 item. The last item in both lists is the token being tab
                 completed
                 **On Failure**
                 - Two empty lists
        """
    import copy
    unclosed_quote = ''
    quotes_to_try = copy.copy(constants.QUOTES)
    tmp_line = line[:endidx]
    tmp_endidx = endidx
    while True:
        try:
            initial_tokens = shlex_split(tmp_line[:tmp_endidx])
            if not unclosed_quote and begidx == tmp_endidx:
                initial_tokens.append('')
            break
        except ValueError as ex:
            if str(ex) == 'No closing quotation' and quotes_to_try:
                unclosed_quote = quotes_to_try[0]
                quotes_to_try = quotes_to_try[1:]
                tmp_line = line[:endidx]
                tmp_line += unclosed_quote
                tmp_endidx = endidx + 1
            else:
                return ([], [])
    raw_tokens = self.statement_parser.split_on_punctuation(initial_tokens)
    tokens = [utils.strip_quotes(cur_token) for cur_token in raw_tokens]
    if unclosed_quote:
        raw_tokens[-1] = raw_tokens[-1][:-1]
    return (tokens, raw_tokens)