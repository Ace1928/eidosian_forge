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
def _adapt_backend_error(cls, err, hash=None, self=None):
    """
        internal helper invoked when backend has hash/verification error;
        used to adapt to passlib message.
        """
    backend = cls.get_backend()
    if self is None and hash is not None:
        self = cls.from_string(hash)
    if self is not None:
        self._validate_constraints(self.memory_cost, self.parallelism)
        if backend == 'argon2_cffi' and self.data is not None:
            raise NotImplementedError("argon2_cffi backend doesn't support the 'data' parameter")
    text = str(err)
    if text not in ['Decoding failed']:
        reason = '%s reported: %s: hash=%r' % (backend, text, hash)
    else:
        reason = repr(hash)
    raise exc.MalformedHashError(cls, reason=reason)