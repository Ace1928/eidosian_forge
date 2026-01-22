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
def _get_ssl_context(self, req: ClientRequest) -> Optional[SSLContext]:
    """Logic to get the correct SSL context

        0. if req.ssl is false, return None

        1. if ssl_context is specified in req, use it
        2. if _ssl_context is specified in self, use it
        3. otherwise:
            1. if verify_ssl is not specified in req, use self.ssl_context
               (will generate a default context according to self.verify_ssl)
            2. if verify_ssl is True in req, generate a default SSL context
            3. if verify_ssl is False in req, generate a SSL context that
               won't verify
        """
    if req.is_ssl():
        if ssl is None:
            raise RuntimeError('SSL is not supported.')
        sslcontext = req.ssl
        if isinstance(sslcontext, ssl.SSLContext):
            return sslcontext
        if sslcontext is not True:
            return self._make_ssl_context(False)
        sslcontext = self._ssl
        if isinstance(sslcontext, ssl.SSLContext):
            return sslcontext
        if sslcontext is not True:
            return self._make_ssl_context(False)
        return self._make_ssl_context(True)
    else:
        return None