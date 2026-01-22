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
def _read_command_line(self, prompt: str) -> str:
    """
        Read command line from appropriate stdin

        :param prompt: prompt to display to user
        :return: command line text of 'eof' if an EOFError was caught
        :raises: whatever exceptions are raised by input() except for EOFError
        """
    try:
        try:
            self.terminal_lock.release()
        except RuntimeError:
            pass
        return self.read_input(prompt, completion_mode=utils.CompletionMode.COMMANDS)
    except EOFError:
        return 'eof'
    finally:
        self.terminal_lock.acquire()