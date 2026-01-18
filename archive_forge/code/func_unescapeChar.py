from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def unescapeChar(self, charcode):
    return chr(int(charcode[1:], 16))