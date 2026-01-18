import os.path
import re
import unittest
import idna
def unicode_fixup(string):
    """Replace backslash-u-XXXX with appropriate unicode characters."""
    return _RE_SURROGATE.sub(lambda match: chr((ord(match.group(0)[0]) - 55296) * 1024 + ord(match.group(0)[1]) - 56320 + 65536), _RE_UNICODE.sub(lambda match: chr(int(match.group(1), 16)), string))