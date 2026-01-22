from __future__ import with_statement, absolute_import
import logging
import re
import types
from warnings import warn
from passlib import exc
from passlib.crypto.digest import MAX_UINT32
from passlib.utils import classproperty, to_bytes, render_bytes
from passlib.utils.binary import b64s_encode, b64s_decode
from passlib.utils.compat import u, unicode, bascii_to_str, uascii_to_str, PY2
import passlib.utils.handlers as uh
@classmethod
def _get_backend_type(cls, value):
    """
        helper to resolve backend constant from type
        """
    try:
        return cls._backend_type_map[value]
    except KeyError:
        pass
    msg = 'unsupported argon2 hash (type %r not supported by %s backend)' % (value, cls.get_backend())
    raise ValueError(msg)