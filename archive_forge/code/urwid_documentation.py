import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
Process the tab key being hit.

        If the line is blank or has only whitespace: indent.

        If there is text before the cursor: cycle completions.

        If `back` is True cycle backwards through completions, and return
        instead of indenting.

        Returns True if the key was handled.
        