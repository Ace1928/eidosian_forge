from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def ulabel(label: Union[str, bytes, bytearray]) -> str:
    if not isinstance(label, (bytes, bytearray)):
        try:
            label_bytes = label.encode('ascii')
        except UnicodeEncodeError:
            check_label(label)
            return label
    else:
        label_bytes = label
    label_bytes = label_bytes.lower()
    if label_bytes.startswith(_alabel_prefix):
        label_bytes = label_bytes[len(_alabel_prefix):]
        if not label_bytes:
            raise IDNAError('Malformed A-label, no Punycode eligible content found')
        if label_bytes.decode('ascii')[-1] == '-':
            raise IDNAError('A-label must not end with a hyphen')
    else:
        check_label(label_bytes)
        return label_bytes.decode('ascii')
    try:
        label = label_bytes.decode('punycode')
    except UnicodeError:
        raise IDNAError('Invalid A-label')
    check_label(label)
    return label