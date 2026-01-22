import asyncio
import functools
import random
import sys
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import suppress
from http import HTTPStatus
from http.cookies import SimpleCookie
from itertools import cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
import attr
from . import hdrs, helpers
from .abc import AbstractResolver
from .client_exceptions import (
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import ceil_timeout, get_running_loop, is_ip_address, noop, sentinel
from .locks import EventResultOrError
from .resolver import DefaultResolver
def _check_loop_for_start_tls(self) -> None:
    try:
        self._loop.start_tls
    except AttributeError as attr_exc:
        raise RuntimeError('An HTTPS request is being sent through an HTTPS proxy. This needs support for TLS in TLS but it is not implemented in your runtime for the stdlib asyncio.\n\nPlease upgrade to Python 3.11 or higher. For more details, please see:\n* https://bugs.python.org/issue37179\n* https://github.com/python/cpython/pull/28073\n* https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support\n* https://github.com/aio-libs/aiohttp/discussions/6044\n') from attr_exc