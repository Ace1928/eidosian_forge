from . import idnadata
import bisect
import unicodedata
import re
from typing import Union, Optional
from .intranges import intranges_contain
def valid_string_length(label: Union[bytes, str], trailing_dot: bool) -> bool:
    if len(label) > (254 if trailing_dot else 253):
        return False
    return True