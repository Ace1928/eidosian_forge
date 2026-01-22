from binascii import hexlify
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_unicode, right_pad_string
from passlib.utils.compat import unicode
from passlib.crypto.digest import lookup_hash
import passlib.utils.handlers as uh
class nthash(uh.StaticHandler):
    """This class implements the NT Password hash, and follows the :ref:`password-hash-api`.

    It has no salt and a single fixed round.

    The :meth:`~passlib.ifc.PasswordHash.hash` and :meth:`~passlib.ifc.PasswordHash.genconfig` methods accept no optional keywords.

    Note that while this class outputs lower-case hexadecimal digests,
    it will accept upper-case digests as well.
    """
    name = 'nthash'
    checksum_chars = uh.HEX_CHARS
    checksum_size = 32

    @classmethod
    def _norm_hash(cls, hash):
        return hash.lower()

    def _calc_checksum(self, secret):
        return hexlify(self.raw(secret)).decode('ascii')

    @classmethod
    def raw(cls, secret):
        """encode password using MD4-based NTHASH algorithm

        :arg secret: secret as unicode or utf-8 encoded bytes

        :returns: returns string of raw bytes
        """
        secret = to_unicode(secret, 'utf-8', param='secret')
        return md4(secret.encode('utf-16-le')).digest()

    @classmethod
    def raw_nthash(cls, secret, hex=False):
        warn('nthash.raw_nthash() is deprecated, and will be removed in Passlib 1.8, please use nthash.raw() instead', DeprecationWarning)
        ret = nthash.raw(secret)
        return hexlify(ret).decode('ascii') if hex else ret