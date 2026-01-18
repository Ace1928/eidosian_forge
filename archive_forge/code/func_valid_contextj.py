from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def valid_contextj(label: str, pos: int) -> bool:
    cp_value = ord(label[pos])
    if cp_value == 8204:
        if pos > 0:
            if _combining_class(ord(label[pos - 1])) == _virama_combining_class:
                return True
        ok = False
        for i in range(pos - 1, -1, -1):
            joining_type = idnadata.joining_types.get(ord(label[i]))
            if joining_type == ord('T'):
                continue
            if joining_type in [ord('L'), ord('D')]:
                ok = True
                break
        if not ok:
            return False
        ok = False
        for i in range(pos + 1, len(label)):
            joining_type = idnadata.joining_types.get(ord(label[i]))
            if joining_type == ord('T'):
                continue
            if joining_type in [ord('R'), ord('D')]:
                ok = True
                break
        return ok
    if cp_value == 8205:
        if pos > 0:
            if _combining_class(ord(label[pos - 1])) == _virama_combining_class:
                return True
        return False
    else:
        return False