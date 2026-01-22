import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def _unescape_safe_chars(matchobj):
    """re.sub callback to convert hex-escapes to plain characters (if safe).

    e.g. '%7E' will be converted to '~'.
    """
    hex_digits = matchobj.group(0)[1:]
    char = chr(int(hex_digits, 16))
    if char in _url_dont_escape_characters:
        return char
    else:
        return matchobj.group(0).upper()