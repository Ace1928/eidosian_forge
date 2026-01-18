from __future__ import annotations
import re
from typing import Match, TypeVar
from .entities import entities
def unescapeAll(string: str) -> str:

    def replacer_func(match: Match[str]) -> str:
        escaped = match.group(1)
        if escaped:
            return escaped
        entity = match.group(2)
        return replaceEntityPattern(match.group(), entity)
    if '\\' not in string and '&' not in string:
        return string
    return UNESCAPE_ALL_RE.sub(replacer_func, string)