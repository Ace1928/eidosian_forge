from __future__ import annotations
import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote
from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError
@staticmethod
def _get_func_code(code: CodeType, name: str) -> t.Callable[..., tuple[str, str]]:
    globs: dict[str, t.Any] = {}
    locs: dict[str, t.Any] = {}
    exec(code, globs, locs)
    return locs[name]