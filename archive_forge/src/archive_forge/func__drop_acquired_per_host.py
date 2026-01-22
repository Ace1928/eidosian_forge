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
def _drop_acquired_per_host(self, key: 'ConnectionKey', val: ResponseHandler) -> None:
    acquired_per_host = self._acquired_per_host
    if key not in acquired_per_host:
        return
    conns = acquired_per_host[key]
    conns.remove(val)
    if not conns:
        del self._acquired_per_host[key]