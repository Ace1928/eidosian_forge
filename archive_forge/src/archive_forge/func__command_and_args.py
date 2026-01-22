import re
import shlex
from typing import (
import attr
from . import (
from .exceptions import (
@staticmethod
def _command_and_args(tokens: List[str]) -> Tuple[str, str]:
    """Given a list of tokens, return a tuple of the command
        and the args as a string.
        """
    command = ''
    args = ''
    if tokens:
        command = tokens[0]
    if len(tokens) > 1:
        args = ' '.join(tokens[1:])
    return (command, args)