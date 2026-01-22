from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
def _encode_field(self, value, param='field'):
    """convert field to internal representation.

        internal representation is always bytes. byte strings are left as-is,
        unicode strings encoding using file's default encoding (or ``utf-8``
        if no encoding has been specified).

        :raises UnicodeEncodeError:
            if unicode value cannot be encoded using default encoding.

        :raises ValueError:
            if resulting byte string contains a forbidden character,
            or is too long (>255 bytes).

        :returns:
            encoded identifer as bytes
        """
    if isinstance(value, unicode):
        value = value.encode(self.encoding)
    elif not isinstance(value, bytes):
        raise ExpectedStringError(value, param)
    if len(value) > 255:
        raise ValueError('%s must be at most 255 characters: %r' % (param, value))
    if any((c in _INVALID_FIELD_CHARS for c in value)):
        raise ValueError('%s contains invalid characters: %r' % (param, value))
    return value