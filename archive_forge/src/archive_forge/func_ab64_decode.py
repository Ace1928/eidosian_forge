from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def ab64_decode(data):
    """
    decode from shortened base64 format which omits padding & whitespace.
    uses custom ``./`` altchars, but supports decoding normal ``+/`` altchars as well.

    it is primarily used by Passlib's custom pbkdf2 hashes.
    """
    if isinstance(data, unicode):
        try:
            data = data.encode('ascii')
        except UnicodeEncodeError:
            raise suppress_cause(ValueError('string argument should contain only ASCII characters'))
    return b64s_decode(data.replace(b'.', b'+'))