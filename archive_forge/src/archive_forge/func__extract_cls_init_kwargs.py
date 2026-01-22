import os
import sys
import logging
import asyncio
import typing
import contextlib
import threading
import inspect
from itertools import chain
from queue import Empty, Full, Queue, LifoQueue
from urllib.parse import parse_qs, unquote, urlparse, ParseResult
from redis.connection import (
from redis.asyncio.connection import (
import aiokeydb.v2.exceptions as exceptions
from aiokeydb.v2.utils import set_ulimits
def _extract_cls_init_kwargs(obj: object) -> typing.List[str]:
    """
    Extracts the kwargs that are valid for a connection class
    """
    argspec = inspect.getfullargspec(obj.__init__)
    _args = []
    for arg in argspec.args:
        _args.append(arg)
    for arg in argspec.kwonlyargs:
        _args.append(arg)
    if hasattr(obj, '__bases__'):
        for base in obj.__bases__:
            _args.extend(_extract_cls_init_kwargs(base))
    return list(set(_args))