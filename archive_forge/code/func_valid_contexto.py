from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def valid_contexto(label: str, pos: int, exception: bool=False) -> bool:
    cp_value = ord(label[pos])
    if cp_value == 183:
        if 0 < pos < len(label) - 1:
            if ord(label[pos - 1]) == 108 and ord(label[pos + 1]) == 108:
                return True
        return False
    elif cp_value == 885:
        if pos < len(label) - 1 and len(label) > 1:
            return _is_script(label[pos + 1], 'Greek')
        return False
    elif cp_value == 1523 or cp_value == 1524:
        if pos > 0:
            return _is_script(label[pos - 1], 'Hebrew')
        return False
    elif cp_value == 12539:
        for cp in label:
            if cp == '・':
                continue
            if _is_script(cp, 'Hiragana') or _is_script(cp, 'Katakana') or _is_script(cp, 'Han'):
                return True
        return False
    elif 1632 <= cp_value <= 1641:
        for cp in label:
            if 1776 <= ord(cp) <= 1785:
                return False
        return True
    elif 1776 <= cp_value <= 1785:
        for cp in label:
            if 1632 <= ord(cp) <= 1641:
                return False
        return True
    return False