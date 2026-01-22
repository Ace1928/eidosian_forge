import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
class TemplateInterpolation(InterpolationEngine):
    """Behaves like string.Template."""
    _cookie = '$'
    _delimiter = '$'
    _KEYCRE = re.compile('\n        \\$(?:\n          (?P<escaped>\\$)              |   # Two $ signs\n          (?P<named>[_a-z][_a-z0-9]*)  |   # $name format\n          {(?P<braced>[^}]*)}              # ${name} format\n        )\n        ', re.IGNORECASE | re.VERBOSE)

    def _parse_match(self, match):
        key = match.group('named') or match.group('braced')
        if key is not None:
            value, section = self._fetch(key)
            return (key, value, section)
        if match.group('escaped') is not None:
            return (None, self._delimiter, None)
        return (None, match.group(), None)