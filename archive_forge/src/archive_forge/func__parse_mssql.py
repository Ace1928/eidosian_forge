from binascii import hexlify, unhexlify
from hashlib import sha1
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import consteq
from passlib.utils.compat import bascii_to_str, unicode, u
import passlib.utils.handlers as uh
def _parse_mssql(hash, csize, bsize, handler):
    """common parser for mssql 2000/2005; returns 4 byte salt + checksum"""
    if isinstance(hash, unicode):
        if len(hash) == csize and hash.startswith(UIDENT):
            try:
                return unhexlify(hash[6:].encode('utf-8'))
            except TypeError:
                pass
    elif isinstance(hash, bytes):
        assert isinstance(hash, bytes)
        if len(hash) == csize and hash.startswith(BIDENT):
            try:
                return unhexlify(hash[6:])
            except TypeError:
                pass
    else:
        raise uh.exc.ExpectedStringError(hash, 'hash')
    raise uh.exc.InvalidHashError(handler)