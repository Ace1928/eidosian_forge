from __future__ import annotations
from functools import lru_cache
import collections
import enum
import os
import re
import typing as T
@classmethod
@lru_cache(maxsize=None)
def _can_dedup(cls, arg: str) -> Dedup:
    """Returns whether the argument can be safely de-duped.

        In addition to these, we handle library arguments specially.
        With GNU ld, we surround library arguments with -Wl,--start/end-group
        to recursively search for symbols in the libraries. This is not needed
        with other linkers.
        """
    if arg in cls.dedup2_prefixes:
        return Dedup.NO_DEDUP
    if arg in cls.dedup2_args or arg.startswith(cls.dedup2_prefixes) or arg.endswith(cls.dedup2_suffixes):
        return Dedup.OVERRIDDEN
    if arg in cls.dedup1_args or arg.startswith(cls.dedup1_prefixes) or arg.endswith(cls.dedup1_suffixes) or re.search(cls.dedup1_regex, arg):
        return Dedup.UNIQUE
    return Dedup.NO_DEDUP