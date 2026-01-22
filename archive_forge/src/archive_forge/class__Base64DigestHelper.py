from base64 import b64encode, b64decode
from hashlib import md5, sha1, sha256, sha512
import logging; log = logging.getLogger(__name__)
import re
from passlib.handlers.misc import plaintext
from passlib.utils import unix_crypt_schemes, to_unicode
from passlib.utils.compat import uascii_to_str, unicode, u
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
class _Base64DigestHelper(uh.StaticHandler):
    """helper for ldap_md5 / ldap_sha1"""
    ident = None
    _hash_func = None
    _hash_regex = None
    checksum_chars = uh.PADDED_BASE64_CHARS

    @classproperty
    def _hash_prefix(cls):
        """tell StaticHandler to strip ident from checksum"""
        return cls.ident

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        chk = self._hash_func(secret).digest()
        return b64encode(chk).decode('ascii')