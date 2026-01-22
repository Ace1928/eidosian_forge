import unicodedata
import enchant.tokenize
def _consume_alpha_b(self, text, offset):
    """Consume an alphabetic character from the given bytestring.

        Given a bytestring and the current offset, this method returns
        the number of characters occupied by the next alphabetic character
        in the string.  Non-ASCII bytes are interpreted as utf-8 and can
        result in multiple characters being consumed.
        """
    assert offset < len(text)
    if text[offset].isalpha():
        return 1
    elif text[offset] >= '\x80':
        return self._consume_alpha_utf8(text, offset)
    return 0