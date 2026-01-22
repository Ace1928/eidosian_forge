from __future__ import absolute_import
import re
import sys
class UnicodeLiteralBuilder(object):
    """Assemble a unicode string.
    """

    def __init__(self):
        self.chars = []

    def append(self, characters):
        if isinstance(characters, _bytes):
            characters = characters.decode('ASCII')
        assert isinstance(characters, _unicode), str(type(characters))
        self.chars.append(characters)
    if sys.maxunicode == 65535:

        def append_charval(self, char_number):
            if char_number > 65535:
                char_number -= 65536
                self.chars.append(_unichr(char_number // 1024 + 55296))
                self.chars.append(_unichr(char_number % 1024 + 56320))
            else:
                self.chars.append(_unichr(char_number))
    else:

        def append_charval(self, char_number):
            self.chars.append(_unichr(char_number))

    def append_uescape(self, char_number, escape_string):
        self.append_charval(char_number)

    def getstring(self):
        return EncodedString(u''.join(self.chars))

    def getstrings(self):
        return (None, self.getstring())