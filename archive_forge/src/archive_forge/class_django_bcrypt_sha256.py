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
class django_bcrypt_sha256(_wrapped_bcrypt):
    """This class implements Django 1.6's Bcrypt+SHA256 hash, and follows the :ref:`password-hash-api`.

    It supports a variable-length salt, and a variable number of rounds.

    While the algorithm and format is somewhat different,
    the api and options for this hash are identical to :class:`!bcrypt` itself,
    see :doc:`bcrypt </lib/passlib.hash.bcrypt>` for more details.

    .. versionadded:: 1.6.2
    """
    name = 'django_bcrypt_sha256'
    django_name = 'bcrypt_sha256'
    _digest = sha256
    django_prefix = u('bcrypt_sha256$')

    @classmethod
    def identify(cls, hash):
        hash = uh.to_unicode_for_identify(hash)
        if not hash:
            return False
        return hash.startswith(cls.django_prefix)

    @classmethod
    def from_string(cls, hash):
        hash = to_unicode(hash, 'ascii', 'hash')
        if not hash.startswith(cls.django_prefix):
            raise uh.exc.InvalidHashError(cls)
        bhash = hash[len(cls.django_prefix):]
        if not bhash.startswith('$2'):
            raise uh.exc.MalformedHashError(cls)
        return super(django_bcrypt_sha256, cls).from_string(bhash)

    def to_string(self):
        bhash = super(django_bcrypt_sha256, self).to_string()
        return uascii_to_str(self.django_prefix) + bhash

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        secret = hexlify(self._digest(secret).digest())
        return super(django_bcrypt_sha256, self)._calc_checksum(secret)