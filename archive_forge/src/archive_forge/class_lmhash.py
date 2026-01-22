from binascii import hexlify
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_unicode, right_pad_string
from passlib.utils.compat import unicode
from passlib.crypto.digest import lookup_hash
import passlib.utils.handlers as uh
class lmhash(uh.TruncateMixin, uh.HasEncodingContext, uh.StaticHandler):
    """This class implements the Lan Manager Password hash, and follows the :ref:`password-hash-api`.

    It has no salt and a single fixed round.

    The :meth:`~passlib.ifc.PasswordHash.using` method accepts a single
    optional keyword:

    :param bool truncate_error:
        By default, this will silently truncate passwords larger than 14 bytes.
        Setting ``truncate_error=True`` will cause :meth:`~passlib.ifc.PasswordHash.hash`
        to raise a :exc:`~passlib.exc.PasswordTruncateError` instead.

        .. versionadded:: 1.7

    The :meth:`~passlib.ifc.PasswordHash.hash` and :meth:`~passlib.ifc.PasswordHash.verify` methods accept a single
    optional keyword:

    :type encoding: str
    :param encoding:

        This specifies what character encoding LMHASH should use when
        calculating digest. It defaults to ``cp437``, the most
        common encoding encountered.

    Note that while this class outputs digests in lower-case hexadecimal,
    it will accept upper-case as well.
    """
    name = 'lmhash'
    setting_kwds = ('truncate_error',)
    checksum_chars = uh.HEX_CHARS
    checksum_size = 32
    truncate_size = 14
    default_encoding = 'cp437'

    @classmethod
    def _norm_hash(cls, hash):
        return hash.lower()

    def _calc_checksum(self, secret):
        if self.use_defaults:
            self._check_truncate_policy(secret)
        return hexlify(self.raw(secret, self.encoding)).decode('ascii')
    _magic = b'KGS!@#$%'

    @classmethod
    def raw(cls, secret, encoding=None):
        """encode password using LANMAN hash algorithm.

        :type secret: unicode or utf-8 encoded bytes
        :arg secret: secret to hash
        :type encoding: str
        :arg encoding:
            optional encoding to use for unicode inputs.
            this defaults to ``cp437``, which is the
            common case for most situations.

        :returns: returns string of raw bytes
        """
        if not encoding:
            encoding = cls.default_encoding
        from passlib.crypto.des import des_encrypt_block
        MAGIC = cls._magic
        if isinstance(secret, unicode):
            secret = secret.upper().encode(encoding)
        elif isinstance(secret, bytes):
            secret = secret.upper()
        else:
            raise TypeError('secret must be unicode or bytes')
        secret = right_pad_string(secret, 14)
        return des_encrypt_block(secret[0:7], MAGIC) + des_encrypt_block(secret[7:14], MAGIC)