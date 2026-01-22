import re
import codecs
from typing import Tuple
from encodings.utf_8 import (
def _buffer_decode_step(self, input, errors, final):
    """
        There are three possibilities for each decoding step:

        - Decode as much real UTF-8 as possible.
        - Decode a six-byte CESU-8 sequence at the current position.
        - Decode a Java-style null at the current position.

        This method figures out which step is appropriate, and does it.
        """
    sup = UTF8IncrementalDecoder._buffer_decode
    match = SPECIAL_BYTES_RE.search(input)
    if match is None:
        return sup(input, errors, final)
    cutoff = match.start()
    if cutoff > 0:
        return sup(input[:cutoff], errors, True)
    if input.startswith(b'\xc0'):
        if len(input) > 1:
            return ('\x00', 2)
        elif final:
            return sup(input, errors, True)
        else:
            return ('', 0)
    else:
        return self._buffer_decode_surrogates(sup, input, errors, final)