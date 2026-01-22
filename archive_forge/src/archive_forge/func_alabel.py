from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def alabel(label: str) -> bytes:
    try:
        label_bytes = label.encode('ascii')
        ulabel(label_bytes)
        if not valid_label_length(label_bytes):
            raise IDNAError('Label too long')
        return label_bytes
    except UnicodeEncodeError:
        pass
    if not label:
        raise IDNAError('No Input')
    label = str(label)
    check_label(label)
    label_bytes = _punycode(label)
    label_bytes = _alabel_prefix + label_bytes
    if not valid_label_length(label_bytes):
        raise IDNAError('Label too long')
    return label_bytes