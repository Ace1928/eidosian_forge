from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def tokenize_rule(s: str) -> list[tuple[str, str]]:
    s = s.split('@')[0]
    result: list[tuple[str, str]] = []
    pos = 0
    end = len(s)
    while pos < end:
        for tok, rule in _RULES:
            match = rule.match(s, pos)
            if match is not None:
                pos = match.end()
                if tok:
                    result.append((tok, match.group()))
                break
        else:
            raise RuleError('malformed CLDR pluralization rule.  Got unexpected %r' % s[pos])
    return result[::-1]