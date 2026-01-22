from __future__ import annotations
from collections import deque
from functools import partial
from io import BytesIO
from time import time
from kombu.asynchronous.hub import READ, WRITE, Hub, get_event_loop
from kombu.exceptions import HttpError
from kombu.utils.encoding import bytes_to_str
from .base import BaseClient
def _push_to_hub(self):
    for fd, events in self._fds.items():
        if events & READ:
            self.hub.add_reader(fd, self.on_readable, fd)
        if events & WRITE:
            self.hub.add_writer(fd, self.on_writable, fd)