import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
class ConfigParserInterpolation(InterpolationEngine):
    """Behaves like ConfigParser."""
    _cookie = '%'
    _KEYCRE = re.compile('%\\(([^)]*)\\)s')

    def _parse_match(self, match):
        key = match.group(1)
        value, section = self._fetch(key)
        return (key, value, section)