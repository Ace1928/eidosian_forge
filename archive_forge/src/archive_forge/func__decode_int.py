from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def _decode_int(self, source, bits):
    """decode base64 string -> integer

        :arg source: base64 string to decode.
        :arg bits: number of bits in resulting integer.

        :raises ValueError:
            * if the string contains invalid base64 characters.
            * if the string is not long enough - it must be at least
              ``int(ceil(bits/6))`` in length.

        :returns:
            a integer in the range ``0 <= n < 2**bits``
        """
    if not isinstance(source, bytes):
        raise TypeError('source must be bytes, not %s' % (type(source),))
    big = self.big
    pad = -bits % 6
    chars = (bits + pad) / 6
    if len(source) != chars:
        raise ValueError('source must be %d chars' % (chars,))
    decode = self._decode64
    out = 0
    try:
        for c in source if big else reversed(source):
            out = (out << 6) + decode(c)
    except KeyError:
        raise ValueError('invalid character in string: %r' % (c,))
    if pad:
        if big:
            out >>= pad
        else:
            out &= (1 << bits) - 1
    return out