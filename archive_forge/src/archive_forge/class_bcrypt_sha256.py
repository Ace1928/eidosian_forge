from __future__ import with_statement, absolute_import
from base64 import b64encode
from hashlib import sha256
import os
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.crypto.digest import compile_hmac
from passlib.exc import PasslibHashWarning, PasslibSecurityWarning, PasslibSecurityError
from passlib.utils import safe_crypt, repeat_string, to_bytes, parse_version, \
from passlib.utils.binary import bcrypt64
from passlib.utils.compat import get_unbound_method_function
from passlib.utils.compat import u, uascii_to_str, unicode, str_to_uascii, PY3, error_from
import passlib.utils.handlers as uh
class bcrypt_sha256(_wrapped_bcrypt):
    """
    This class implements a composition of BCrypt + HMAC_SHA256,
    and follows the :ref:`password-hash-api`.

    It supports a fixed-length salt, and a variable number of rounds.

    The :meth:`~passlib.ifc.PasswordHash.hash` and :meth:`~passlib.ifc.PasswordHash.genconfig` methods accept
    all the same optional keywords as the base :class:`bcrypt` hash.

    .. versionadded:: 1.6.2

    .. versionchanged:: 1.7

        Now defaults to ``"2b"`` bcrypt variant; though supports older hashes
        generated using the ``"2a"`` bcrypt variant.

    .. versionchanged:: 1.7.3

        For increased security, updated to use HMAC-SHA256 instead of plain SHA256.
        Now only supports the ``"2b"`` bcrypt variant.  Hash format updated to "v=2".
    """
    name = 'bcrypt_sha256'
    ident_values = (IDENT_2A, IDENT_2B)
    ident_aliases = (lambda ident_values: dict((item for item in bcrypt.ident_aliases.items() if item[1] in ident_values)))(ident_values)
    default_ident = IDENT_2B
    _supported_versions = set([1, 2])
    version = 2

    @classmethod
    def using(cls, version=None, **kwds):
        subcls = super(bcrypt_sha256, cls).using(**kwds)
        if version is not None:
            subcls.version = subcls._norm_version(version)
        ident = subcls.default_ident
        if subcls.version > 1 and ident != IDENT_2B:
            raise ValueError('bcrypt %r hashes not allowed for version %r' % (ident, subcls.version))
        return subcls
    prefix = u('$bcrypt-sha256$')
    _v2_hash_re = re.compile('(?x)\n        ^\n        [$]bcrypt-sha256[$]\n        v=(?P<version>\\d+),\n        t=(?P<type>2b),\n        r=(?P<rounds>\\d{1,2})\n        [$](?P<salt>[^$]{22})\n        (?:[$](?P<digest>[^$]{31}))?\n        $\n        ')
    _v1_hash_re = re.compile('(?x)\n        ^\n        [$]bcrypt-sha256[$]\n        (?P<type>2[ab]),\n        (?P<rounds>\\d{1,2})\n        [$](?P<salt>[^$]{22})\n        (?:[$](?P<digest>[^$]{31}))?\n        $\n        ')

    @classmethod
    def identify(cls, hash):
        hash = uh.to_unicode_for_identify(hash)
        if not hash:
            return False
        return hash.startswith(cls.prefix)

    @classmethod
    def from_string(cls, hash):
        hash = to_unicode(hash, 'ascii', 'hash')
        if not hash.startswith(cls.prefix):
            raise uh.exc.InvalidHashError(cls)
        m = cls._v2_hash_re.match(hash)
        if m:
            version = int(m.group('version'))
            if version < 2:
                raise uh.exc.MalformedHashError(cls)
        else:
            m = cls._v1_hash_re.match(hash)
            if m:
                version = 1
            else:
                raise uh.exc.MalformedHashError(cls)
        rounds = m.group('rounds')
        if rounds.startswith(uh._UZERO) and rounds != uh._UZERO:
            raise uh.exc.ZeroPaddedRoundsError(cls)
        return cls(version=version, ident=m.group('type'), rounds=int(rounds), salt=m.group('salt'), checksum=m.group('digest'))
    _v2_template = u('$bcrypt-sha256$v=2,t=%s,r=%d$%s$%s')
    _v1_template = u('$bcrypt-sha256$%s,%d$%s$%s')

    def to_string(self):
        if self.version == 1:
            template = self._v1_template
        else:
            template = self._v2_template
        hash = template % (self.ident.strip(_UDOLLAR), self.rounds, self.salt, self.checksum)
        return uascii_to_str(hash)

    def __init__(self, version=None, **kwds):
        if version is not None:
            self.version = self._norm_version(version)
        super(bcrypt_sha256, self).__init__(**kwds)

    @classmethod
    def _norm_version(cls, version):
        if version not in cls._supported_versions:
            raise ValueError('%s: unknown or unsupported version: %r' % (cls.name, version))
        return version

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        if self.version == 1:
            digest = sha256(secret).digest()
        else:
            salt = self.salt
            if salt[-1] not in self.final_salt_chars:
                raise ValueError('invalid salt string')
            digest = compile_hmac('sha256', salt.encode('ascii'))(secret)
        key = b64encode(digest)
        return super(bcrypt_sha256, self)._calc_checksum(key)

    def _calc_needs_update(self, **kwds):
        if self.version < type(self).version:
            return True
        return super(bcrypt_sha256, self)._calc_needs_update(**kwds)