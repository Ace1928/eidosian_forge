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
@staticmethod
def _is_path_match(req_path: str, cookie_path: str) -> bool:
    """Implements path matching adhering to RFC 6265."""
    if not req_path.startswith('/'):
        req_path = '/'
    if req_path == cookie_path:
        return True
    if not req_path.startswith(cookie_path):
        return False
    if cookie_path.endswith('/'):
        return True
    non_matching = req_path[len(cookie_path):]
    return non_matching.startswith('/')