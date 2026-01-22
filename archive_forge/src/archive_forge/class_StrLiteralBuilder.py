from __future__ import absolute_import
import re
import sys
class StrLiteralBuilder(object):
    """Assemble both a bytes and a unicode representation of a string.
    """

    def __init__(self, target_encoding):
        self._bytes = BytesLiteralBuilder(target_encoding)
        self._unicode = UnicodeLiteralBuilder()

    def append(self, characters):
        self._bytes.append(characters)
        self._unicode.append(characters)

    def append_charval(self, char_number):
        self._bytes.append_charval(char_number)
        self._unicode.append_charval(char_number)

    def append_uescape(self, char_number, escape_string):
        self._bytes.append(escape_string)
        self._unicode.append_charval(char_number)

    def getstrings(self):
        return (self._bytes.getstring(), self._unicode.getstring())