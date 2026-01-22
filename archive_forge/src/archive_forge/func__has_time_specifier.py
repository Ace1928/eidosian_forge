from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _has_time_specifier(date_format: str) -> bool:
    i = 0
    length = len(date_format)
    while i < length:
        if date_format[i] == '%':
            i += 1
            if i < length and date_format[i] in TIME_SPECIFIERS:
                return True
        i += 1
    return False