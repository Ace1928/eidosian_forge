from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
class _PasslibHasherWrapper(object):
    """
    adapter which which wraps a :cls:`passlib.ifc.PasswordHash` class,
    and provides an interface compatible with the Django hasher API.

    :param passlib_handler:
        passlib hash handler (e.g. :cls:`passlib.hash.sha256_crypt`.
    """
    passlib_handler = None

    def __init__(self, passlib_handler):
        if getattr(passlib_handler, 'django_name', None):
            raise ValueError("handlers that reflect an official django hasher shouldn't be wrapped: %r" % (passlib_handler.name,))
        if passlib_handler.is_disabled:
            raise ValueError("can't wrap disabled-hash handlers: %r" % passlib_handler.name)
        self.passlib_handler = passlib_handler
        if self._has_rounds:
            self.rounds = passlib_handler.default_rounds
            self.iterations = ProxyProperty('rounds')

    def __repr__(self):
        return '<PasslibHasherWrapper handler=%r>' % self.passlib_handler

    @memoized_property
    def __name__(self):
        return 'Passlib_%s_PasswordHasher' % self.passlib_handler.name.title()

    @memoized_property
    def _has_rounds(self):
        return 'rounds' in self.passlib_handler.setting_kwds

    @memoized_property
    def _translate_kwds(self):
        """
        internal helper for safe_summary() --
        used to translate passlib hash options -> django keywords
        """
        out = dict(checksum='hash')
        if self._has_rounds and 'pbkdf2' in self.passlib_handler.name:
            out['rounds'] = 'iterations'
        return out

    @memoized_property
    def algorithm(self):
        return PASSLIB_WRAPPER_PREFIX + self.passlib_handler.name

    def salt(self):
        return _GEN_SALT_SIGNAL

    def verify(self, password, encoded):
        return self.passlib_handler.verify(password, encoded)

    def encode(self, password, salt=None, rounds=None, iterations=None):
        kwds = {}
        if salt is not None and salt != _GEN_SALT_SIGNAL:
            kwds['salt'] = salt
        if self._has_rounds:
            if rounds is not None:
                kwds['rounds'] = rounds
            elif iterations is not None:
                kwds['rounds'] = iterations
            else:
                kwds['rounds'] = self.rounds
        elif rounds is not None or iterations is not None:
            warn("%s.hash(): 'rounds' and 'iterations' are ignored" % self.__name__)
        handler = self.passlib_handler
        if kwds:
            handler = handler.using(**kwds)
        return handler.hash(password)

    def safe_summary(self, encoded):
        from django.contrib.auth.hashers import mask_hash
        from django.utils.translation import ugettext_noop as _
        handler = self.passlib_handler
        items = [(_('algorithm'), handler.name)]
        if hasattr(handler, 'parsehash'):
            kwds = handler.parsehash(encoded, sanitize=mask_hash)
            for key, value in iteritems(kwds):
                key = self._translate_kwds.get(key, key)
                items.append((_(key), value))
        return OrderedDict(items)

    def must_update(self, encoded):
        if self._has_rounds:
            subcls = self.passlib_handler.using(min_rounds=self.rounds, max_rounds=self.rounds)
            if subcls.needs_update(encoded):
                return True
        return False