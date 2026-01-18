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
def to_asyncio_future(tornado_future: asyncio.Future) -> asyncio.Future:
    """Convert a Tornado yieldable object to an `asyncio.Future`.

    .. versionadded:: 4.1

    .. versionchanged:: 4.3
       Now accepts any yieldable object, not just
       `tornado.concurrent.Future`.

    .. deprecated:: 5.0
       Tornado ``Futures`` have been merged with `asyncio.Future`,
       so this method is now equivalent to `tornado.gen.convert_yielded`.
    """
    return convert_yielded(tornado_future)