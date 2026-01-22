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
def _fail_on_no_start_tls(self, req: 'ClientRequest') -> None:
    """Raise a :py:exc:`RuntimeError` on missing ``start_tls()``.

        It is necessary for TLS-in-TLS so that it is possible to
        send HTTPS queries through HTTPS proxies.

        This doesn't affect regular HTTP requests, though.
        """
    if not req.is_ssl():
        return
    proxy_url = req.proxy
    assert proxy_url is not None
    if proxy_url.scheme != 'https':
        return
    self._check_loop_for_start_tls()