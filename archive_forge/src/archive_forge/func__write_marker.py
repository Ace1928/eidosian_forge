import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _write_marker(self, indent_string, depth, entry, comment):
    """Write a section marker line"""
    return '%s%s%s%s%s' % (indent_string, self._a_to_u('[' * depth), self._quote(self._decode_element(entry), multiline=False), self._a_to_u(']' * depth), self._decode_element(comment))