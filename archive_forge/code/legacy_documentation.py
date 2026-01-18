import warnings
from typing import Dict, Optional, Union
from .api import from_bytes, from_fp, from_path, normalize
from .constant import CHARDET_CORRESPONDENCE
from .models import CharsetMatch, CharsetMatches

    chardet legacy method
    Detect the encoding of the given byte string. It should be mostly backward-compatible.
    Encoding name will match Chardet own writing whenever possible. (Not on encoding name unsupported by it)
    This function is deprecated and should be used to migrate your project easily, consult the documentation for
    further information. Not planned for removal.

    :param byte_str:     The byte sequence to examine.
    