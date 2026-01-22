from __future__ import annotations
import html
import itertools
import re
import unicodedata
def _build_regexes():
    """
    ENCODING_REGEXES contain reasonably fast ways to detect if we
    could represent a given string in a given encoding. The simplest one is
    the 'ascii' detector, which of course just determines if all characters
    are between U+0000 and U+007F.
    """
    encoding_regexes = {'ascii': re.compile('^[\x00-\x7f]*$')}
    for encoding in CHARMAP_ENCODINGS:
        byte_range = bytes(list(range(128, 256)) + [26])
        charlist = byte_range.decode(encoding)
        regex = '^[\x00-\x19\x1b-\x7f{0}]*$'.format(charlist)
        encoding_regexes[encoding] = re.compile(regex)
    return encoding_regexes