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
def using_standout_or_underline(a: AttrSpec | str) -> bool:
    a = self._pal_attrspec.get(a, a)
    return isinstance(a, AttrSpec) and (a.standout or a.underline)