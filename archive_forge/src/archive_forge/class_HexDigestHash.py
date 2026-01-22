import hashlib
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_native_str, to_bytes, render_bytes, consteq
from passlib.utils.compat import unicode, str_to_uascii
import passlib.utils.handlers as uh
from passlib.crypto.digest import lookup_hash
class HexDigestHash(uh.StaticHandler):
    """this provides a template for supporting passwords stored as plain hexadecimal hashes"""
    _hash_func = None
    checksum_size = None
    checksum_chars = uh.HEX_CHARS
    supported = True

    @classmethod
    def _norm_hash(cls, hash):
        return hash.lower()

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        return str_to_uascii(self._hash_func(secret).hexdigest())