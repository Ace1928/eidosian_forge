import base64
import binascii
import re
import string
import six
def WebSafeBase64Escape(unescaped, do_padding):
    """Python implementation of the Google C library's WebSafeBase64Escape().

  Python implementation of the Google C library's WebSafeBase64Escape() (from
  strings/strutil.h), using Python's base64 API and string replacement.

  Args:
    unescaped: any data (byte) string (example: b"12345~6")
    do_padding: whether to add =-padding (example: false)

  Returns:
    The base64 encoding (with web-safe replacements) of unescaped,
    with =-padding depending on the value of do_padding
    (example: b"MTIzNDV-Ng")
  """
    escaped = base64.urlsafe_b64encode(unescaped)
    if not do_padding:
        escaped = escaped.rstrip(b'=')
    return escaped