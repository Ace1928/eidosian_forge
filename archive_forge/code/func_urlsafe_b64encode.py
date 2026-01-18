import re
import struct
import binascii
def urlsafe_b64encode(s):
    """Encode bytes using the URL- and filesystem-safe Base64 alphabet.

    Argument s is a bytes-like object to encode.  The result is returned as a
    bytes object.  The alphabet uses '-' instead of '+' and '_' instead of
    '/'.
    """
    return b64encode(s).translate(_urlsafe_encode_translation)