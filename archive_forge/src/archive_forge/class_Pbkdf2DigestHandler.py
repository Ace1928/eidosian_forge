from binascii import hexlify, unhexlify
from base64 import b64encode, b64decode
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_unicode
from passlib.utils.binary import ab64_decode, ab64_encode
from passlib.utils.compat import str_to_bascii, u, uascii_to_str, unicode
from passlib.crypto.digest import pbkdf2_hmac
import passlib.utils.handlers as uh
class Pbkdf2DigestHandler(uh.HasRounds, uh.HasRawSalt, uh.HasRawChecksum, uh.GenericHandler):
    """base class for various pbkdf2_{digest} algorithms"""
    setting_kwds = ('salt', 'salt_size', 'rounds')
    checksum_chars = uh.HASH64_CHARS
    default_salt_size = 16
    max_salt_size = 1024
    default_rounds = None
    min_rounds = 1
    max_rounds = 4294967295
    rounds_cost = 'linear'
    _digest = None

    @classmethod
    def from_string(cls, hash):
        rounds, salt, chk = uh.parse_mc3(hash, cls.ident, handler=cls)
        salt = ab64_decode(salt.encode('ascii'))
        if chk:
            chk = ab64_decode(chk.encode('ascii'))
        return cls(rounds=rounds, salt=salt, checksum=chk)

    def to_string(self):
        salt = ab64_encode(self.salt).decode('ascii')
        chk = ab64_encode(self.checksum).decode('ascii')
        return uh.render_mc3(self.ident, self.rounds, salt, chk)

    def _calc_checksum(self, secret):
        return pbkdf2_hmac(self._digest, secret, self.salt, self.rounds, self.checksum_size)