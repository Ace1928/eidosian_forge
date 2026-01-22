import os
import re
import sys
from codecs import BOM_UTF8, BOM_UTF16, BOM_UTF16_BE, BOM_UTF16_LE
import six
from ._version import __version__
def _decode_element(self, line):
    """Decode element to unicode if necessary."""
    if isinstance(line, six.binary_type) and self.default_encoding:
        return line.decode(self.default_encoding)
    else:
        return line