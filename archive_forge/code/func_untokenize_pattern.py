from __future__ import annotations
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, SupportsInt
import datetime
from collections.abc import Iterable
from babel import localtime
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def untokenize_pattern(tokens: Iterable[tuple[str, str | tuple[str, int]]]) -> str:
    """
    Turn a date format pattern token stream back into a string.

    This is the reverse operation of ``tokenize_pattern``.

    :type tokens: Iterable[tuple]
    :rtype: str
    """
    output = []
    for tok_type, tok_value in tokens:
        if tok_type == 'field':
            output.append(tok_value[0] * tok_value[1])
        elif tok_type == 'chars':
            if not any((ch in PATTERN_CHARS for ch in tok_value)):
                output.append(tok_value)
            else:
                output.append("'%s'" % tok_value.replace("'", "''"))
    return ''.join(output)