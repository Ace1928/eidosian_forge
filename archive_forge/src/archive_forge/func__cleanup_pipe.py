from __future__ import annotations
import dataclasses
import glob
import html
import os
import pathlib
import random
import selectors
import signal
import socket
import string
import sys
import tempfile
import typing
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width, move_next_char
from urwid.util import StoppingContext, get_encoding
from .common import BaseScreen
def _cleanup_pipe(self, *args) -> None:
    if not self.pipe_name:
        return
    with suppress(Exception):
        os.remove(f'{self.pipe_name}.in')
        os.remove(f'{self.pipe_name}.update')