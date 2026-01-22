from __future__ import annotations
import dataclasses
import enum
import functools
import inspect
from inspect import Parameter
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Final
from typing import NoReturn
import py
def ascii_escaped(val: bytes | str) -> str:
    """If val is pure ASCII, return it as an str, otherwise, escape
    bytes objects into a sequence of escaped bytes:

    b'\\xc3\\xb4\\xc5\\xd6' -> r'\\xc3\\xb4\\xc5\\xd6'

    and escapes strings into a sequence of escaped unicode ids, e.g.:

    r'4\\nV\\U00043efa\\x0eMXWB\\x1e\\u3028\\u15fd\\xcd\\U0007d944'

    Note:
       The obvious "v.decode('unicode-escape')" will return
       valid UTF-8 unicode if it finds them in bytes, but we
       want to return escaped bytes for any byte, even if they match
       a UTF-8 string.
    """
    if isinstance(val, bytes):
        ret = val.decode('ascii', 'backslashreplace')
    else:
        ret = val.encode('unicode_escape').decode('ascii')
    return ret.translate(_non_printable_ascii_translate_table)