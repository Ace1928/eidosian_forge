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
def _fork_child(self) -> None:
    """
        Fork a child to run CGI disconnected for polling update method.
        Force parent process to exit.
        """
    daemonize(f'{self.pipe_name}.err')
    self.input_fd = os.open(f'{self.pipe_name}.in', os.O_NONBLOCK | os.O_RDONLY)
    self.update_method = 'polling child'
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(f'{self.pipe_name}.update')
    s.listen(1)
    s.settimeout(POLL_CONNECT)
    self.server_socket = s