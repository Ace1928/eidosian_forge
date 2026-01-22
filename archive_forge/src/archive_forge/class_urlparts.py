from __future__ import annotations
from collections.abc import Mapping
from functools import partial
from typing import NamedTuple
from urllib.parse import parse_qsl, quote, unquote, urlparse
from ..log import get_logger
class urlparts(NamedTuple):
    """Named tuple representing parts of the URL."""
    scheme: str
    hostname: str
    port: int
    username: str
    password: str
    path: str
    query: Mapping