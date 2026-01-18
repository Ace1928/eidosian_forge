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
@classproperty
def type_values(cls):
    """
        return tuple of types supported by this backend
        
        .. versionadded:: 1.7.2
        """
    cls.get_backend()
    return tuple(cls._backend_type_map)