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
def _reset_timer(self):
    """Reset the timer from message."""
    if self.timer is not None:
        self.main_loop.remove_alarm(self.timer)
        self.timer = None