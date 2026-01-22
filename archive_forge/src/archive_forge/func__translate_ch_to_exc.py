import time of Click down, some infrequently used functionality is
import contextlib
import math
import os
import sys
import time
import typing as t
from gettext import gettext as _
from io import StringIO
from types import TracebackType
from ._compat import _default_text_stdout
from ._compat import CYGWIN
from ._compat import get_best_encoding
from ._compat import isatty
from ._compat import open_stream
from ._compat import strip_ansi
from ._compat import term_len
from ._compat import WIN
from .exceptions import ClickException
from .utils import echo
def _translate_ch_to_exc(ch: str) -> t.Optional[BaseException]:
    if ch == '\x03':
        raise KeyboardInterrupt()
    if ch == '\x04' and (not WIN):
        raise EOFError()
    if ch == '\x1a' and WIN:
        raise EOFError()
    return None