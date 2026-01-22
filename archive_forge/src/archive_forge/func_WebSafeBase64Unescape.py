import base64
import binascii
import re
import string
import six
def WebSafeBase64Unescape(escaped):
    """Python implementation of the Google C library's WebSafeBase64Unescape().

  Python implementation of the Google C library's WebSafeBase64Unescape() (from
  strings/strutil.h), using Python's base64 API and string replacement.

  Args:
    escaped: A base64 binary string using the web-safe encoding
        (example: b"MTIzNDV-Ng")

  Returns:
    The corresponding unescaped string (example: b"12345~6")

  Raises:
    Base64ValueError: Invalid character in encoding of string, escaped.
  """
    escaped_standard = escaped.translate(_BASE64_DECODE_TRANSLATION)
    if escaped_standard.find(b'!') >= 0:
        raise Base64ValueError('%r: Invalid character in encoded string.' % escaped)
    if not escaped_standard.endswith(b'='):
        padding_len = len(escaped_standard) % 4
        escaped_standard += b'=' * padding_len
    try:
        return binascii.a2b_base64(escaped_standard)
    except binascii.Error as msg:
        raise Base64ValueError('%r: %s' % (escaped, msg))