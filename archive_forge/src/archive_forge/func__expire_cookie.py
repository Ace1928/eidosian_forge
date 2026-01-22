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
def _expire_cookie(self, when: float, domain: str, path: str, name: str) -> None:
    self._next_expiration = min(self._next_expiration, when)
    self._expirations[domain, path, name] = when