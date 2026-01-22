from __future__ import with_statement
import inspect
import logging; log = logging.getLogger(__name__)
import math
import threading
from warnings import warn
import passlib.exc as exc, passlib.ifc as ifc
from passlib.exc import MissingBackendError, PasslibConfigWarning, \
from passlib.ifc import PasswordHash
from passlib.registry import get_crypt_handler
from passlib.utils import (
from passlib.utils.binary import (
from passlib.utils.compat import join_byte_values, irange, u, native_string_types, \
from passlib.utils.decor import classproperty, deprecated_method
class PrefixWrapper(object):
    """wraps another handler, adding a constant prefix.

    instances of this class wrap another password hash handler,
    altering the constant prefix that's prepended to the wrapped
    handlers' hashes.

    this is used mainly by the :doc:`ldap crypt <passlib.hash.ldap_crypt>` handlers;
    such as :class:`~passlib.hash.ldap_md5_crypt` which wraps :class:`~passlib.hash.md5_crypt` and adds a ``{CRYPT}`` prefix.

    usage::

        myhandler = PrefixWrapper("myhandler", "md5_crypt", prefix="$mh$", orig_prefix="$1$")

    :param name: name to assign to handler
    :param wrapped: handler object or name of registered handler
    :param prefix: identifying prefix to prepend to all hashes
    :param orig_prefix: prefix to strip (defaults to '').
    :param lazy: if True and wrapped handler is specified by name, don't look it up until needed.
    """
    _using_clone_attrs = ()

    def __init__(self, name, wrapped, prefix=u(''), orig_prefix=u(''), lazy=False, doc=None, ident=None):
        self.name = name
        if isinstance(prefix, bytes):
            prefix = prefix.decode('ascii')
        self.prefix = prefix
        if isinstance(orig_prefix, bytes):
            orig_prefix = orig_prefix.decode('ascii')
        self.orig_prefix = orig_prefix
        if doc:
            self.__doc__ = doc
        if hasattr(wrapped, 'name'):
            self._set_wrapped(wrapped)
        else:
            self._wrapped_name = wrapped
            if not lazy:
                self._get_wrapped()
        if ident is not None:
            if ident is True:
                if prefix:
                    ident = prefix
                else:
                    raise ValueError('no prefix specified')
            if isinstance(ident, bytes):
                ident = ident.decode('ascii')
            if ident[:len(prefix)] != prefix[:len(ident)]:
                raise ValueError('ident must agree with prefix')
            self._ident = ident
    _wrapped_name = None
    _wrapped_handler = None

    def _set_wrapped(self, handler):
        if 'ident' in handler.setting_kwds and self.orig_prefix:
            warn("PrefixWrapper: 'orig_prefix' option may not work correctly for handlers which have multiple identifiers: %r" % (handler.name,), exc.PasslibRuntimeWarning)
        self._wrapped_handler = handler

    def _get_wrapped(self):
        handler = self._wrapped_handler
        if handler is None:
            handler = get_crypt_handler(self._wrapped_name)
            self._set_wrapped(handler)
        return handler
    wrapped = property(_get_wrapped)
    _ident = False

    @property
    def ident(self):
        value = self._ident
        if value is False:
            value = None
            if not self.orig_prefix:
                wrapped = self.wrapped
                ident = getattr(wrapped, 'ident', None)
                if ident is not None:
                    value = self._wrap_hash(ident)
            self._ident = value
        return value
    _ident_values = False

    @property
    def ident_values(self):
        value = self._ident_values
        if value is False:
            value = None
            if not self.orig_prefix:
                wrapped = self.wrapped
                idents = getattr(wrapped, 'ident_values', None)
                if idents:
                    value = tuple((self._wrap_hash(ident) for ident in idents))
            self._ident_values = value
        return value
    _proxy_attrs = ('setting_kwds', 'context_kwds', 'default_rounds', 'min_rounds', 'max_rounds', 'rounds_cost', 'min_desired_rounds', 'max_desired_rounds', 'vary_rounds', 'default_salt_size', 'min_salt_size', 'max_salt_size', 'salt_chars', 'default_salt_chars', 'backends', 'has_backend', 'get_backend', 'set_backend', 'is_disabled', 'truncate_size', 'truncate_error', 'truncate_verify_reject', '_salt_is_bytes')

    def __repr__(self):
        args = [repr(self._wrapped_name or self._wrapped_handler)]
        if self.prefix:
            args.append('prefix=%r' % self.prefix)
        if self.orig_prefix:
            args.append('orig_prefix=%r' % self.orig_prefix)
        args = ', '.join(args)
        return 'PrefixWrapper(%r, %s)' % (self.name, args)

    def __dir__(self):
        attrs = set(dir(self.__class__))
        attrs.update(self.__dict__)
        wrapped = self.wrapped
        attrs.update((attr for attr in self._proxy_attrs if hasattr(wrapped, attr)))
        return list(attrs)

    def __getattr__(self, attr):
        """proxy most attributes from wrapped class (e.g. rounds, salt size, etc)"""
        if attr in self._proxy_attrs:
            return getattr(self.wrapped, attr)
        raise AttributeError('missing attribute: %r' % (attr,))

    def __setattr__(self, attr, value):
        if attr in self._proxy_attrs and self._derived_from:
            wrapped = self.wrapped
            if hasattr(wrapped, attr):
                setattr(wrapped, attr, value)
                return
        return object.__setattr__(self, attr, value)

    def _unwrap_hash(self, hash):
        """given hash belonging to wrapper, return orig version"""
        prefix = self.prefix
        if not hash.startswith(prefix):
            raise exc.InvalidHashError(self)
        return self.orig_prefix + hash[len(prefix):]

    def _wrap_hash(self, hash):
        """given orig hash; return one belonging to wrapper"""
        if isinstance(hash, bytes):
            hash = hash.decode('ascii')
        orig_prefix = self.orig_prefix
        if not hash.startswith(orig_prefix):
            raise exc.InvalidHashError(self.wrapped)
        wrapped = self.prefix + hash[len(orig_prefix):]
        return uascii_to_str(wrapped)
    _derived_from = None

    def using(self, **kwds):
        subcls = self.wrapped.using(**kwds)
        assert subcls is not self.wrapped
        wrapper = PrefixWrapper(self.name, subcls, prefix=self.prefix, orig_prefix=self.orig_prefix)
        wrapper._derived_from = self
        for attr in self._using_clone_attrs:
            setattr(wrapper, attr, getattr(self, attr))
        return wrapper

    def needs_update(self, hash, **kwds):
        hash = self._unwrap_hash(hash)
        return self.wrapped.needs_update(hash, **kwds)

    def identify(self, hash):
        hash = to_unicode_for_identify(hash)
        if not hash.startswith(self.prefix):
            return False
        hash = self._unwrap_hash(hash)
        return self.wrapped.identify(hash)

    @deprecated_method(deprecated='1.7', removed='2.0')
    def genconfig(self, **kwds):
        config = self.wrapped.genconfig(**kwds)
        if config is None:
            raise RuntimeError('.genconfig() must return a string, not None')
        return self._wrap_hash(config)

    @deprecated_method(deprecated='1.7', removed='2.0')
    def genhash(self, secret, config, **kwds):
        if config is not None:
            config = to_unicode(config, 'ascii', 'config/hash')
            config = self._unwrap_hash(config)
        return self._wrap_hash(self.wrapped.genhash(secret, config, **kwds))

    @deprecated_method(deprecated='1.7', removed='2.0', replacement='.hash()')
    def encrypt(self, secret, **kwds):
        return self.hash(secret, **kwds)

    def hash(self, secret, **kwds):
        return self._wrap_hash(self.wrapped.hash(secret, **kwds))

    def verify(self, secret, hash, **kwds):
        hash = to_unicode(hash, 'ascii', 'hash')
        hash = self._unwrap_hash(hash)
        return self.wrapped.verify(secret, hash, **kwds)