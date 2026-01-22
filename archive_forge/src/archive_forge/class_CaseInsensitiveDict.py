import asyncio
import datetime
import hashlib
import inspect
import re
import time as mod_time
import warnings
from typing import (
from aioredis.compat import Protocol, TypedDict
from aioredis.connection import (
from aioredis.exceptions import (
from aioredis.lock import Lock
from aioredis.utils import safe_str, str_if_bytes
class CaseInsensitiveDict(dict):
    """Case insensitive dict implementation. Assumes string keys only."""

    def __init__(self, data):
        for k, v in data.items():
            self[k.upper()] = v

    def __contains__(self, k):
        return super().__contains__(k.upper())

    def __delitem__(self, k):
        super().__delitem__(k.upper())

    def __getitem__(self, k):
        return super().__getitem__(k.upper())

    def get(self, k, default=None):
        return super().get(k.upper(), default)

    def __setitem__(self, k, v):
        super().__setitem__(k.upper(), v)

    def update(self, data):
        data = CaseInsensitiveDict(data)
        super().update(data)