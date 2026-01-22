from __future__ import annotations
from collections.abc import MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING
@lru_cache(maxsize=256)
def _compile_eval(source):
    """
    Cached compile in eval mode
    """
    return compile(source, '<string-expression>', 'eval')