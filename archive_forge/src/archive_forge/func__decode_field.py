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
def _decode_field(self, value):
    """decode field from internal representation to format
        returns by users() method, etc.

        :raises UnicodeDecodeError:
            if unicode value cannot be decoded using default encoding.
            (usually indicates wrong encoding set for file).

        :returns:
            field as unicode or bytes, as appropriate.
        """
    assert isinstance(value, bytes), 'expected value to be bytes'
    if self.return_unicode:
        return value.decode(self.encoding)
    else:
        return value