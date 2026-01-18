import abc
import code
import inspect
import os
import pkgutil
import pydoc
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from abc import abstractmethod
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from types import ModuleType, TracebackType
from typing import (
from ._typing_compat import Literal
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from . import autocomplete, inspection, simpleeval
from .config import getpreferredencoding, Config
from .formatter import Parenthesis
from .history import History
from .lazyre import LazyReCompile
from .paste import PasteHelper, PastePinnwand, PasteFailed
from .patch_linecache import filename_for_console_input
from .translations import _, ngettext
from .importcompletion import ModuleGatherer
def write2file(self) -> None:
    """Prompt for a filename and write the current contents of the stdout
        buffer to disk."""
    try:
        fn = self.interact.file_prompt(_('Save to file (Esc to cancel): '))
        if not fn:
            self.interact.notify(_('Save cancelled.'))
            return
    except ValueError:
        self.interact.notify(_('Save cancelled.'))
        return
    path = Path(fn).expanduser()
    if path.suffix != '.py' and self.config.save_append_py:
        path = Path(f'{path}.py')
    mode = 'w'
    if path.exists():
        new_mode = self.interact.file_prompt(_('%s already exists. Do you want to (c)ancel, (o)verwrite or (a)ppend? ') % (path,))
        if new_mode in ('o', 'overwrite', _('overwrite')):
            mode = 'w'
        elif new_mode in ('a', 'append', _('append')):
            mode = 'a'
        else:
            self.interact.notify(_('Save cancelled.'))
            return
    stdout_text = self.get_session_formatted_for_file()
    try:
        with open(path, mode) as f:
            f.write(stdout_text)
    except OSError as e:
        self.interact.notify(_("Error writing file '%s': %s") % (path, e))
    else:
        self.interact.notify(_('Saved to %s.') % (path,))