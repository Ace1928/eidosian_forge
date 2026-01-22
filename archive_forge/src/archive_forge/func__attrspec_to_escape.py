from __future__ import annotations
import abc
import contextlib
import functools
import os
import platform
import selectors
import signal
import socket
import sys
import typing
from urwid import signals, str_util, util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, UPDATE_PALETTE_ENTRY, AttrSpec, BaseScreen, RealTerminal
def _attrspec_to_escape(self, a: AttrSpec) -> str:
    """
        Convert AttrSpec instance a to an escape sequence for the terminal

        >>> s = Screen()
        >>> s.set_terminal_properties(colors=256)
        >>> a2e = s._attrspec_to_escape
        >>> a2e(s.AttrSpec('brown', 'dark green'))
        '\\x1b[0;33;42m'
        >>> a2e(s.AttrSpec('#fea,underline', '#d0d'))
        '\\x1b[0;38;5;229;4;48;5;164m'
        """
    if self.term == 'fbterm':
        fg = escape.ESC + f'[1;{a.foreground_number:d}}}'
        bg = escape.ESC + f'[2;{a.background_number:d}}}'
        return fg + bg
    if a.foreground_true:
        fg = f'38;2;{';'.join((str(part) for part in a.get_rgb_values()[0:3]))}'
    elif a.foreground_high:
        fg = f'38;5;{a.foreground_number:d}'
    elif a.foreground_basic:
        if a.foreground_number > 7:
            if self.fg_bright_is_bold:
                fg = f'1;{a.foreground_number - 8 + 30:d}'
            else:
                fg = f'{a.foreground_number - 8 + 90:d}'
        else:
            fg = f'{a.foreground_number + 30:d}'
    else:
        fg = '39'
    st = '1;' * a.bold + '3;' * a.italics + '4;' * a.underline + '5;' * a.blink + '7;' * a.standout + '9;' * a.strikethrough
    if a.background_true:
        bg = f'48;2;{';'.join((str(part) for part in a.get_rgb_values()[3:6]))}'
    elif a.background_high:
        bg = f'48;5;{a.background_number:d}'
    elif a.background_basic:
        if a.background_number > 7:
            if self.bg_bright_is_blink:
                bg = f'5;{a.background_number - 8 + 40:d}'
            else:
                bg = f'{a.background_number - 8 + 100:d}'
        else:
            bg = f'{a.background_number + 40:d}'
    else:
        bg = '49'
    return f'{escape.ESC}[0;{fg};{st}{bg}m'