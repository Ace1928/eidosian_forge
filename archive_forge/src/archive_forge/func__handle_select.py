import asyncio
import atexit
import concurrent.futures
import errno
import functools
import select
import socket
import sys
import threading
import typing
import warnings
from tornado.gen import convert_yielded
from tornado.ioloop import IOLoop, _Selectable
from typing import (
def _handle_select(self, rs: List[_FileDescriptorLike], ws: List[_FileDescriptorLike]) -> None:
    for r in rs:
        self._handle_event(r, self._readers)
    for w in ws:
        self._handle_event(w, self._writers)
    self._start_select()