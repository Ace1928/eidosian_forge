from binascii import hexlify, unhexlify
from hashlib import md5
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import right_pad_string, to_unicode, repeat_string, to_bytes
from passlib.utils.binary import h64
from passlib.utils.compat import unicode, u, join_byte_values, \
import passlib.utils.handlers as uh
class cisco_pix(uh.HasUserContext, uh.StaticHandler):
    """
    This class implements the password hash used by older Cisco PIX firewalls,
    and follows the :ref:`password-hash-api`.
    It does a single round of hashing, and relies on the username
    as the salt.

    This class only allows passwords <= 16 bytes, anything larger
    will result in a :exc:`~passlib.exc.PasswordSizeError` if passed to :meth:`~cisco_pix.hash`,
    and be silently rejected if passed to :meth:`~cisco_pix.verify`.

    The :meth:`~passlib.ifc.PasswordHash.hash`,
    :meth:`~passlib.ifc.PasswordHash.genhash`, and
    :meth:`~passlib.ifc.PasswordHash.verify` methods
    all support the following extra keyword:

    :param str user:
        String containing name of user account this password is associated with.

        This is *required* in order to correctly hash passwords associated
        with a user account on the Cisco device, as it is used to salt
        the hash.

        Conversely, this *must* be omitted or set to ``""`` in order to correctly
        hash passwords which don't have an associated user account
        (such as the "enable" password).

    .. versionadded:: 1.6

    .. versionchanged:: 1.7.1

        Passwords > 16 bytes are now rejected / throw error instead of being silently truncated,
        to match Cisco behavior.  A number of :ref:`bugs <passlib-asa96-bug>` were fixed
        which caused prior releases to generate unverifiable hashes in certain cases.
    """
    name = 'cisco_pix'
    truncate_size = 16
    truncate_error = True
    truncate_verify_reject = True
    checksum_size = 16
    checksum_chars = uh.HASH64_CHARS
    _is_asa = False

    def _calc_checksum(self, secret):
        """
        This function implements the "encrypted" hash format used by Cisco
        PIX & ASA. It's behavior has been confirmed for ASA 9.6,
        but is presumed correct for PIX & other ASA releases,
        as it fits with known test vectors, and existing literature.

        While nearly the same, the PIX & ASA hashes have slight differences,
        so this function performs differently based on the _is_asa class flag.
        Noteable changes from PIX to ASA include password size limit
        increased from 16 -> 32, and other internal changes.
        """
        asa = self._is_asa
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        spoil_digest = None
        if len(secret) > self.truncate_size:
            if self.use_defaults:
                msg = 'Password too long (%s allows at most %d bytes)' % (self.name, self.truncate_size)
                raise uh.exc.PasswordSizeError(self.truncate_size, msg=msg)
            else:
                spoil_digest = secret + _DUMMY_BYTES
        user = self.user
        if user:
            if isinstance(user, unicode):
                user = user.encode('utf-8')
            if not asa or len(secret) < 28:
                secret += repeat_string(user, 4)
        if asa and len(secret) > 16:
            pad_size = 32
        else:
            pad_size = 16
        secret = right_pad_string(secret, pad_size)
        if spoil_digest:
            secret += spoil_digest
        digest = md5(secret).digest()
        digest = join_byte_elems((c for i, c in enumerate(digest) if i + 1 & 3))
        return h64.encode_bytes(digest).decode('ascii')