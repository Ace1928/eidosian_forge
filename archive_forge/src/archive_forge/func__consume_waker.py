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
def _consume_waker(self) -> None:
    try:
        self._waker_r.recv(1024)
    except BlockingIOError:
        pass