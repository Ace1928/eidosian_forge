import abc
import functools
import itertools
import re
import warnings
from typing import (
from .utils import canonicalize_version
from .version import LegacyVersion, Version, parse
def _require_version_compare(fn: Callable[['Specifier', ParsedVersion, str], bool]) -> Callable[['Specifier', ParsedVersion, str], bool]:

    @functools.wraps(fn)
    def wrapped(self: 'Specifier', prospective: ParsedVersion, spec: str) -> bool:
        if not isinstance(prospective, Version):
            return False
        return fn(self, prospective, spec)
    return wrapped