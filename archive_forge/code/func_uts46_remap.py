from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def uts46_remap(domain: str, std3_rules: bool=True, transitional: bool=False) -> str:
    """Re-map the characters in the string according to UTS46 processing."""
    from .uts46data import uts46data
    output = ''
    for pos, char in enumerate(domain):
        code_point = ord(char)
        try:
            uts46row = uts46data[code_point if code_point < 256 else bisect.bisect_left(uts46data, (code_point, 'Z')) - 1]
            status = uts46row[1]
            replacement = None
            if len(uts46row) == 3:
                replacement = uts46row[2]
            if status == 'V' or (status == 'D' and (not transitional)) or (status == '3' and (not std3_rules) and (replacement is None)):
                output += char
            elif replacement is not None and (status == 'M' or (status == '3' and (not std3_rules)) or (status == 'D' and transitional)):
                output += replacement
            elif status != 'I':
                raise IndexError()
        except IndexError:
            raise InvalidCodepoint('Codepoint {} not allowed at position {} in {}'.format(_unot(code_point), pos + 1, repr(domain)))
    return unicodedata.normalize('NFC', output)