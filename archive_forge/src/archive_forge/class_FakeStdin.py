import contextlib
import errno
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from enum import Enum
from types import FrameType, TracebackType
from typing import (
from .._typing_compat import Literal
import greenlet
from curtsies import (
from curtsies.configfile_keynames import keymap as key_dispatch
from curtsies.input import is_main_thread
from curtsies.window import CursorAwareWindow
from cwcwidth import wcswidth
from pygments import format as pygformat
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from . import events as bpythonevents, sitefix, replpainter as paint
from ..config import Config
from .coderunner import (
from .filewatch import ModuleChangedEventHandler
from .interaction import StatusBar
from .interpreter import (
from .manual_readline import (
from .parse import parse as bpythonparse, func_for_letter, color_for_letter
from .preprocess import preprocess
from .. import __version__
from ..config import getpreferredencoding
from ..formatter import BPythonFormatter
from ..pager import get_pager_command
from ..repl import (
from ..translations import _
from ..line import CHARACTER_PAIR_MAP
class FakeStdin:
    """The stdin object user code will reference

    In user code, sys.stdin.read() asks the user for interactive input,
    so this class returns control to the UI to get that input."""

    def __init__(self, coderunner: CodeRunner, repl: 'BaseRepl', configured_edit_keys: Optional[AbstractEdits]=None):
        self.coderunner = coderunner
        self.repl = repl
        self.has_focus = False
        self.current_line = ''
        self.cursor_offset = 0
        self.old_num_lines = 0
        self.readline_results: List[str] = []
        if configured_edit_keys is not None:
            self.rl_char_sequences = configured_edit_keys
        else:
            self.rl_char_sequences = edit_keys

    def process_event(self, e: Union[events.Event, str]) -> None:
        assert self.has_focus
        logger.debug('fake input processing event %r', e)
        if isinstance(e, events.Event):
            if isinstance(e, events.PasteEvent):
                for ee in e.events:
                    if ee not in self.rl_char_sequences:
                        self.add_input_character(ee)
            elif isinstance(e, events.SigIntEvent):
                self.coderunner.sigint_happened_in_main_context = True
                self.has_focus = False
                self.current_line = ''
                self.cursor_offset = 0
                self.repl.run_code_and_maybe_finish()
        elif e in self.rl_char_sequences:
            self.cursor_offset, self.current_line = self.rl_char_sequences[e](self.cursor_offset, self.current_line)
        elif e == '<Ctrl-d>':
            if not len(self.current_line):
                self.repl.send_to_stdin('\n')
                self.has_focus = False
                self.current_line = ''
                self.cursor_offset = 0
                self.repl.run_code_and_maybe_finish(for_code='')
        elif e in ('\n', '\r', '<Ctrl-j>', '<Ctrl-m>'):
            line = f'{self.current_line}\n'
            self.repl.send_to_stdin(line)
            self.has_focus = False
            self.current_line = ''
            self.cursor_offset = 0
            self.repl.run_code_and_maybe_finish(for_code=line)
        elif e != '<ESC>':
            self.add_input_character(e)
        if not self.current_line.endswith(('\n', '\r')):
            self.repl.send_to_stdin(self.current_line)

    def add_input_character(self, e: str) -> None:
        if e == '<SPACE>':
            e = ' '
        if e.startswith('<') and e.endswith('>'):
            return
        assert len(e) == 1, 'added multiple characters: %r' % e
        logger.debug('adding normal char %r to current line', e)
        self.current_line = self.current_line[:self.cursor_offset] + e + self.current_line[self.cursor_offset:]
        self.cursor_offset += 1

    def readline(self, size: int=-1) -> str:
        if not isinstance(size, int):
            raise TypeError(f"'{type(size).__name__}' object cannot be interpreted as an integer")
        elif size == 0:
            return ''
        self.has_focus = True
        self.repl.send_to_stdin(self.current_line)
        value = self.coderunner.request_from_main_context()
        assert isinstance(value, str)
        self.readline_results.append(value)
        return value if size <= -1 else value[:size]

    def readlines(self, size: Optional[int]=-1) -> List[str]:
        if size is None:
            size = -1
        if not isinstance(size, int):
            raise TypeError("argument should be integer or None, not 'str'")
        if size <= 0:
            return list(iter(self.readline, ''))
        lines = []
        while size > 0:
            line = self.readline()
            lines.append(line)
            size -= len(line)
        return lines

    def __iter__(self):
        return iter(self.readlines())

    def isatty(self) -> bool:
        return True

    def flush(self) -> None:
        """Flush the internal buffer. This is a no-op. Flushing stdin
        doesn't make any sense anyway."""

    def write(self, value):
        raise OSError(errno.EBADF, 'sys.stdin is read-only')

    def close(self) -> None:
        pass

    @property
    def encoding(self) -> str:
        return sys.__stdin__.encoding