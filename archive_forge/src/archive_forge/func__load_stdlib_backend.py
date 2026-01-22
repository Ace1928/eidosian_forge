from __future__ import absolute_import
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib import exc
from passlib.utils import to_bytes
from passlib.utils.compat import PYPY
def _load_stdlib_backend():
    """
    Attempt to load stdlib scrypt() implement and return wrapper.
    Returns None if not found.
    """
    try:
        from hashlib import scrypt as stdlib_scrypt
    except ImportError:
        return None

    def stdlib_scrypt_wrapper(secret, salt, n, r, p, keylen):
        maxmem = SCRYPT_MAXMEM
        if maxmem < 0:
            maxmem = estimate_maxmem(n, r, p)
        return stdlib_scrypt(password=secret, salt=salt, n=n, r=r, p=p, dklen=keylen, maxmem=maxmem)
    return stdlib_scrypt_wrapper