from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
@classmethod
def _clip_to_valid_salt_size(cls, salt_size, param='salt_size', relaxed=True):
    """
        internal helper --
        clip salt size value to handler's absolute limits (min_salt_size / max_salt_size)

        :param relaxed:
            if ``True`` (the default), issues PasslibHashWarning is rounds are outside allowed range.
            if ``False``, raises a ValueError instead.

        :param param:
            optional name of parameter to insert into error/warning messages.

        :returns:
            clipped rounds value
        """
    mn = cls.min_salt_size
    mx = cls.max_salt_size
    if mn == mx:
        if salt_size != mn:
            msg = '%s: %s (%d) must be exactly %d' % (cls.name, param, salt_size, mn)
            if relaxed:
                warn(msg, PasslibHashWarning)
            else:
                raise ValueError(msg)
        return mn
    if salt_size < mn:
        msg = '%s: %s (%r) below min_salt_size (%d)' % (cls.name, param, salt_size, mn)
        if relaxed:
            warn(msg, PasslibHashWarning)
            salt_size = mn
        else:
            raise ValueError(msg)
    if mx and salt_size > mx:
        msg = '%s: %s (%r) above max_salt_size (%d)' % (cls.name, param, salt_size, mx)
        if relaxed:
            warn(msg, PasslibHashWarning)
            salt_size = mx
        else:
            raise ValueError(msg)
    return salt_size