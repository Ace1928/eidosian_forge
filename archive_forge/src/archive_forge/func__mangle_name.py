from __future__ import annotations
from typing import Any, Mapping
def _mangle_name(name: str, prefix: str) -> str:
    if name.startswith('__'):
        prefix = '_' + prefix
    else:
        prefix = ''
    return prefix + name