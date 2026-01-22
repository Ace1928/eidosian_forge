import curses
import errno
import functools
import math
import os
import platform
import re
import struct
import sys
import time
from typing import (
from ._typing_compat import Literal
import unicodedata
from dataclasses import dataclass
from pygments import format
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from .formatter import BPythonFormatter
from .config import getpreferredencoding, Config
from .keys import cli_key_dispatch as key_dispatch
from . import translations
from .translations import _
from . import repl, inspection
from . import args as bpargs
from .pager import page
from .args import parse as argsparse
def addstr(self, s: str) -> None:
    """Add a string to the current input line and figure out
        where it should go, depending on the cursor position."""
    self.rl_history.reset()
    if not self.cpos:
        self.s += s
    else:
        l = len(self.s)
        self.s = self.s[:l - self.cpos] + s + self.s[l - self.cpos:]
    self.complete()