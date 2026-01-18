import asyncio
import base64
import binascii
import contextlib
import datetime
import enum
import functools
import inspect
import netrc
import os
import platform
import re
import sys
import time
import warnings
import weakref
from collections import namedtuple
from contextlib import suppress
from email.parser import HeaderParser
from email.utils import parsedate
from math import ceil
from pathlib import Path
from types import TracebackType
from typing import (
from urllib.parse import quote
from urllib.request import getproxies, proxy_bypass
import attr
from multidict import MultiDict, MultiDictProxy, MultiMapping
from yarl import URL
from . import hdrs
from .log import client_logger, internal_logger
def weakref_handle(ob: object, name: str, timeout: float, loop: asyncio.AbstractEventLoop, timeout_ceil_threshold: float=5) -> Optional[asyncio.TimerHandle]:
    if timeout is not None and timeout > 0:
        when = loop.time() + timeout
        if timeout >= timeout_ceil_threshold:
            when = ceil(when)
        return loop.call_at(when, _weakref_handle, (weakref.ref(ob), name))
    return None