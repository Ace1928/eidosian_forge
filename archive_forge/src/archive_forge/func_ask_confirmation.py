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
def ask_confirmation(self, q):
    """Ask for yes or no and return boolean"""
    try:
        reply = self.statusbar.prompt(q)
    except ValueError:
        return False
    return reply.lower() in ('y', 'yes')