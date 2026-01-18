import warnings
from typing import Dict, Optional, Sequence
def unidecode_expect_nonascii(string: str, errors: str='ignore', replace_str: str='?') -> str:
    """Transliterate an Unicode object into an ASCII string

    >>> unidecode("北亰")
    "Bei Jing "

    See unidecode_expect_ascii.
    """
    return _unidecode(string, errors, replace_str)