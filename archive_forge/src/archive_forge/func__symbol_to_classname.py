import sys
import re
from abc import ABCMeta
from unicodedata import name as unicode_name
from decimal import Decimal, DecimalException
from typing import Any, cast, overload, Callable, Dict, Generic, List, \
def _symbol_to_classname(symbol: str) -> str:
    """
    Converts a symbol string to an identifier (only alphanumeric and '_').
    """

    def get_id_name(c: str) -> str:
        if c.isalnum() or c == '_':
            return c
        else:
            return '%s_' % unicode_name(str(c)).title()
    if symbol.isalnum():
        return symbol.title()
    elif symbol in SPECIAL_SYMBOLS:
        return symbol[1:-1].title()
    elif all((c in '-_' for c in symbol)):
        value = ' '.join((unicode_name(c) for c in symbol))
        return value.title().replace(' ', '').replace('-', '').replace('_', '')
    value = symbol.replace('-', '_')
    if value.isidentifier():
        return value.title().replace('_', '')
    value = ''.join((get_id_name(c) for c in symbol))
    return value.replace(' ', '').replace('-', '').replace('_', '')