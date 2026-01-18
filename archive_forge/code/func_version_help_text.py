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
def version_help_text(self) -> str:
    help_message = _("\nThanks for using bpython!\n\nSee http://bpython-interpreter.org/ for more information and http://docs.bpython-interpreter.org/ for docs.\nPlease report issues at https://github.com/bpython/bpython/issues\n\nFeatures:\nTry using undo ({config.undo_key})!\nEdit the current line ({config.edit_current_block_key}) or the entire session ({config.external_editor_key}) in an external editor. (currently {config.editor})\nSave sessions ({config.save_key}) or post them to pastebins ({config.pastebin_key})! Current pastebin helper: {config.pastebin_helper}\nReload all modules and rerun session ({config.reimport_key}) to test out changes to a module.\nToggle auto-reload mode ({config.toggle_file_watch_key}) to re-execute the current session when a module you've imported is modified.\n\nbpython -i your_script.py runs a file in interactive mode\nbpython -t your_script.py pastes the contents of a file into the session\n\nA config file at {config.config_path} customizes keys and behavior of bpython.\nYou can also set which pastebin helper and which external editor to use.\nSee {example_config_url} for an example config file.\nPress {config.edit_config_key} to edit this config file.\n").format(example_config_url=EXAMPLE_CONFIG_URL, config=self.config)
    return f'bpython-curtsies version {__version__} using curtsies version {curtsies_version}\n{help_message}'