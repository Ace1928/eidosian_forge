import abc
import asyncio
import base64
import hashlib
import inspect
import keyword
import os
import re
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import (
from yarl import URL, __version__ as yarl_version  # type: ignore[attr-defined]
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import (
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef
def validation(self, domain: str) -> str:
    if not isinstance(domain, str):
        raise TypeError('Domain must be str')
    domain = domain.rstrip('.').lower()
    if not domain:
        raise ValueError('Domain cannot be empty')
    elif '://' in domain:
        raise ValueError('Scheme not supported')
    url = URL('http://' + domain)
    assert url.raw_host is not None
    if not all((self.re_part.fullmatch(x) for x in url.raw_host.split('.'))):
        raise ValueError('Domain not valid')
    if url.port == 80:
        return url.raw_host
    return f'{url.raw_host}:{url.port}'