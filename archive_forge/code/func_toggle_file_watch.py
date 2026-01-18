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
def toggle_file_watch(self):
    if self.watcher:
        if self.watching_files:
            msg = _('Auto-reloading deactivated.')
            self.status_bar.message(msg)
            self.watcher.deactivate()
            self.watching_files = False
        else:
            msg = _('Auto-reloading active, watching for file changes...')
            self.status_bar.message(msg)
            self.watching_files = True
            self.watcher.activate()
    else:
        self.status_bar.message(_('Auto-reloading not available because watchdog not installed.'))