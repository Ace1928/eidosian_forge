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
class HeadersMixin:
    ATTRS = frozenset(['_content_type', '_content_dict', '_stored_content_type'])
    _headers: MultiMapping[str]
    _content_type: Optional[str] = None
    _content_dict: Optional[Dict[str, str]] = None
    _stored_content_type: Union[str, None, _SENTINEL] = sentinel

    def _parse_content_type(self, raw: Optional[str]) -> None:
        self._stored_content_type = raw
        if raw is None:
            self._content_type = 'application/octet-stream'
            self._content_dict = {}
        else:
            msg = HeaderParser().parsestr('Content-Type: ' + raw)
            self._content_type = msg.get_content_type()
            params = msg.get_params(())
            self._content_dict = dict(params[1:])

    @property
    def content_type(self) -> str:
        """The value of content part for Content-Type HTTP header."""
        raw = self._headers.get(hdrs.CONTENT_TYPE)
        if self._stored_content_type != raw:
            self._parse_content_type(raw)
        return self._content_type

    @property
    def charset(self) -> Optional[str]:
        """The value of charset part for Content-Type HTTP header."""
        raw = self._headers.get(hdrs.CONTENT_TYPE)
        if self._stored_content_type != raw:
            self._parse_content_type(raw)
        return self._content_dict.get('charset')

    @property
    def content_length(self) -> Optional[int]:
        """The value of Content-Length HTTP header."""
        content_length = self._headers.get(hdrs.CONTENT_LENGTH)
        if content_length is not None:
            return int(content_length)
        else:
            return None