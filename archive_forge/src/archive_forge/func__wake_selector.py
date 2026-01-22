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
def _wake_selector(self) -> None:
    if self._closed:
        return
    try:
        self._waker_w.send(b'a')
    except BlockingIOError:
        pass