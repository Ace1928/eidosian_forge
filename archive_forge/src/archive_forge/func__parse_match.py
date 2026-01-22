import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _parse_match(self, match):
    key = match.group('named') or match.group('braced')
    if key is not None:
        value, section = self._fetch(key)
        return (key, value, section)
    if match.group('escaped') is not None:
        return (None, self._delimiter, None)
    return (None, match.group(), None)