from base64 import b64encode
from binascii import hexlify
from hashlib import md5, sha1, sha256
import logging; log = logging.getLogger(__name__)
from passlib.handlers.bcrypt import _wrapped_bcrypt
from passlib.hash import argon2, bcrypt, pbkdf2_sha1, pbkdf2_sha256
from passlib.utils import to_unicode, rng, getrandstr
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import str_to_uascii, uascii_to_str, unicode, u
from passlib.crypto.digest import pbkdf2_hmac
import passlib.utils.handlers as uh
class DjangoSaltedHash(uh.HasSalt, uh.GenericHandler):
    """base class providing common code for django hashes"""
    setting_kwds = ('salt', 'salt_size')
    default_salt_size = 12
    max_salt_size = None
    salt_chars = SALT_CHARS
    checksum_chars = uh.LOWER_HEX_CHARS

    @classmethod
    def from_string(cls, hash):
        salt, chk = uh.parse_mc2(hash, cls.ident, handler=cls)
        return cls(salt=salt, checksum=chk)

    def to_string(self):
        return uh.render_mc2(self.ident, self.salt, self.checksum)