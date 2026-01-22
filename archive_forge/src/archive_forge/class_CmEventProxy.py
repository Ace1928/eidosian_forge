import contextvars
import importlib
import itertools
import json
import logging
import pathlib
import typing
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
import trio
from trio_websocket import ConnectionClosed as WsConnectionClosed
from trio_websocket import connect_websocket_url
@dataclass
class CmEventProxy:
    """A proxy object returned by :meth:`CdpBase.wait_for()``.

    After the context manager executes, this proxy object will have a
    value set that contains the returned event.
    """
    value: typing.Any = None