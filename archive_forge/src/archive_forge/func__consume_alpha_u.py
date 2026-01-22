import unicodedata
import enchant.tokenize
def _consume_alpha_u(self, text, offset):
    """Consume an alphabetic character from the given unicode string.

        Given a unicode string and the current offset, this method returns
        the number of characters occupied by the next alphabetic character
        in the string.  Trailing combining characters are consumed as a
        single letter.
        """
    assert offset < len(text)
    incr = 0
    if text[offset].isalpha():
        incr = 1
        while offset + incr < len(text):
            if unicodedata.category(text[offset + incr])[0] != 'M':
                break
            incr += 1
    return incr