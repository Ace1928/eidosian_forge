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
class BasicAuth(namedtuple('BasicAuth', ['login', 'password', 'encoding'])):
    """Http basic authentication helper."""

    def __new__(cls, login: str, password: str='', encoding: str='latin1') -> 'BasicAuth':
        if login is None:
            raise ValueError('None is not allowed as login value')
        if password is None:
            raise ValueError('None is not allowed as password value')
        if ':' in login:
            raise ValueError('A ":" is not allowed in login (RFC 1945#section-11.1)')
        return super().__new__(cls, login, password, encoding)

    @classmethod
    def decode(cls, auth_header: str, encoding: str='latin1') -> 'BasicAuth':
        """Create a BasicAuth object from an Authorization HTTP header."""
        try:
            auth_type, encoded_credentials = auth_header.split(' ', 1)
        except ValueError:
            raise ValueError('Could not parse authorization header.')
        if auth_type.lower() != 'basic':
            raise ValueError('Unknown authorization method %s' % auth_type)
        try:
            decoded = base64.b64decode(encoded_credentials.encode('ascii'), validate=True).decode(encoding)
        except binascii.Error:
            raise ValueError('Invalid base64 encoding.')
        try:
            username, password = decoded.split(':', 1)
        except ValueError:
            raise ValueError('Invalid credentials.')
        return cls(username, password, encoding=encoding)

    @classmethod
    def from_url(cls, url: URL, *, encoding: str='latin1') -> Optional['BasicAuth']:
        """Create BasicAuth from url."""
        if not isinstance(url, URL):
            raise TypeError('url should be yarl.URL instance')
        if url.user is None:
            return None
        return cls(url.user, url.password or '', encoding=encoding)

    def encode(self) -> str:
        """Encode credentials."""
        creds = f'{self.login}:{self.password}'.encode(self.encoding)
        return 'Basic %s' % base64.b64encode(creds).decode(self.encoding)