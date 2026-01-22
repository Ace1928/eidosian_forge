from __future__ import annotations
import inspect
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Callable
@lru_cache(maxsize=10)
def _get_module_documentation_group(modname) -> str:
    for prefix, group in _module_prefixes:
        if modname.startswith(prefix):
            return group
    raise ValueError(f'No known documentation group for module {modname!r}')