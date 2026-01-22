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
class CLIInteraction(repl.Interaction):

    def __init__(self, config: Config, statusbar: 'Statusbar'):
        super().__init__(config)
        self.statusbar = statusbar

    def confirm(self, q: str) -> bool:
        """Ask for yes or no and return boolean"""
        try:
            reply = self.statusbar.prompt(q)
        except ValueError:
            return False
        return reply.lower() in (_('y'), _('yes'))

    def notify(self, s: str, n: float=10.0, wait_for_keypress: bool=False) -> None:
        self.statusbar.message(s, n)

    def file_prompt(self, s: str) -> Optional[str]:
        return self.statusbar.prompt(s)