import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _write_line(self, indent_string, entry, this_entry, comment):
    """Write an individual line, for the write method"""
    if not self.unrepr:
        val = self._decode_element(self._quote(this_entry))
    else:
        val = repr(this_entry)
    return '%s%s%s%s%s' % (indent_string, self._decode_element(self._quote(entry, multiline=False)), self._a_to_u(' = '), val, self._decode_element(comment))