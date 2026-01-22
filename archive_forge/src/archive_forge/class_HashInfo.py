from __future__ import division
import hashlib
import logging; log = logging.getLogger(__name__)
import re
import os
from struct import Struct
from warnings import warn
from passlib import exc
from passlib.utils import join_bytes, to_native_str, join_byte_values, to_bytes, \
from passlib.utils.compat import irange, int_types, unicode_or_bytes_types, PY3, error_from
from passlib.utils.decor import memoized_property
class HashInfo(SequenceMixin):
    """
    Record containing information about a given hash algorithm, as returned :func:`lookup_hash`.

    This class exposes the following attributes:

    .. autoattribute:: const
    .. autoattribute:: digest_size
    .. autoattribute:: block_size
    .. autoattribute:: name
    .. autoattribute:: iana_name
    .. autoattribute:: aliases
    .. autoattribute:: supported

    This object can also be treated a 3-element sequence
    containing ``(const, digest_size, block_size)``.
    """
    name = None
    iana_name = None
    aliases = ()
    const = None
    digest_size = None
    block_size = None
    error_text = None
    unknown = False

    def __init__(self, const, names, required=True):
        """
        initialize new instance.
        :arg const:
            hash constructor
        :arg names:
            list of 2+ names. should be list of ``(name, iana_name, ... 0+ aliases)``.
            names must be lower-case. only iana name may be None.
        """
        name = self.name = names[0]
        self.iana_name = names[1]
        self.aliases = names[2:]

        def use_stub_const(msg):
            """
            helper that installs stub constructor which throws specified error <msg>.
            """

            def const(source=b''):
                raise exc.UnknownHashError(msg, name)
            if required:
                const()
                assert "shouldn't get here"
            self.error_text = msg
            self.const = const
            try:
                self.digest_size, self.block_size = _fallback_info[name]
            except KeyError:
                pass
        if const is None:
            if names in _known_hash_names:
                msg = 'unsupported hash: %r' % name
            else:
                msg = 'unknown hash: %r' % name
                self.unknown = True
            use_stub_const(msg)
            return
        try:
            hash = const()
        except ValueError as err:
            if 'disabled for fips' in str(err).lower():
                msg = '%r hash disabled for fips' % name
            else:
                msg = 'internal error in %r constructor\n(%s: %s)' % (name, type(err).__name__, err)
            use_stub_const(msg)
            return
        self.const = const
        self.digest_size = hash.digest_size
        self.block_size = hash.block_size
        if len(hash.digest()) != hash.digest_size:
            raise RuntimeError('%r constructor failed sanity check' % self.name)
        if hash.name != self.name:
            warn('inconsistent digest name: %r resolved to %r, which reports name as %r' % (self.name, const, hash.name), exc.PasslibRuntimeWarning)

    def __repr__(self):
        return '<lookup_hash(%r): digest_size=%r block_size=%r)' % (self.name, self.digest_size, self.block_size)

    def _as_tuple(self):
        return (self.const, self.digest_size, self.block_size)

    @memoized_property
    def supported(self):
        """
        whether hash is available for use
        (if False, constructor will throw UnknownHashError if called)
        """
        return self.error_text is None

    @memoized_property
    def supported_by_fastpbkdf2(self):
        """helper to detect if hash is supported by fastpbkdf2()"""
        if not _fast_pbkdf2_hmac:
            return None
        try:
            _fast_pbkdf2_hmac(self.name, b'p', b's', 1)
            return True
        except ValueError:
            return False

    @memoized_property
    def supported_by_hashlib_pbkdf2(self):
        """helper to detect if hash is supported by hashlib.pbkdf2_hmac()"""
        if not _stdlib_pbkdf2_hmac:
            return None
        try:
            _stdlib_pbkdf2_hmac(self.name, b'p', b's', 1)
            return True
        except ValueError:
            return False