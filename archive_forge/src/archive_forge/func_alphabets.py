import warnings
from collections import Counter
from encodings.aliases import aliases
from hashlib import sha256
from json import dumps
from re import sub
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from .constant import NOT_PRINTABLE_PATTERN, TOO_BIG_SEQUENCE
from .md import mess_ratio
from .utils import iana_name, is_multi_byte_encoding, unicode_range
@property
def alphabets(self) -> List[str]:
    if self._unicode_ranges is not None:
        return self._unicode_ranges
    detected_ranges = [unicode_range(char) for char in str(self)]
    self._unicode_ranges = sorted(list({r for r in detected_ranges if r}))
    return self._unicode_ranges