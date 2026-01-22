import asyncio
import calendar
import contextlib
import datetime
import os  # noqa
import pathlib
import pickle
import re
import time
from collections import defaultdict
from http.cookies import BaseCookie, Morsel, SimpleCookie
from math import ceil
from typing import (  # noqa
from yarl import URL
from .abc import AbstractCookieJar, ClearCookiePredicate
from .helpers import is_ip_address
from .typedefs import LooseCookies, PathLike, StrOrURL
class DummyCookieJar(AbstractCookieJar):
    """Implements a dummy cookie storage.

    It can be used with the ClientSession when no cookie processing is needed.

    """

    def __init__(self, *, loop: Optional[asyncio.AbstractEventLoop]=None) -> None:
        super().__init__(loop=loop)

    def __iter__(self) -> 'Iterator[Morsel[str]]':
        while False:
            yield None

    def __len__(self) -> int:
        return 0

    def clear(self, predicate: Optional[ClearCookiePredicate]=None) -> None:
        pass

    def clear_domain(self, domain: str) -> None:
        pass

    def update_cookies(self, cookies: LooseCookies, response_url: URL=URL()) -> None:
        pass

    def filter_cookies(self, request_url: URL) -> 'BaseCookie[str]':
        return SimpleCookie()