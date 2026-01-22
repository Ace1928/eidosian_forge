from __future__ import annotations
import functools
import importlib
import pkgutil
import threading
from typing import Any, Callable, Optional, Sequence
import tiktoken_ext
from tiktoken.core import Encoding
@functools.lru_cache()
def _available_plugin_modules() -> Sequence[str]:
    mods = []
    plugin_mods = pkgutil.iter_modules(tiktoken_ext.__path__, tiktoken_ext.__name__ + '.')
    for _, mod_name, _ in plugin_mods:
        mods.append(mod_name)
    return mods