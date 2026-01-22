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
class DjangoTranslator(object):
    """
    Object which helps translate passlib hasher objects / names
    to and from django hasher objects / names.

    These methods are wrapped in a class so that results can be cached,
    but with the ability to have independant caches, since django hasher
    names may / may not correspond to the same instance (or even class).
    """
    context = None
    _django_hasher_cache = None
    _django_unsalted_sha1 = None
    _passlib_hasher_cache = None

    def __init__(self, context=None, **kwds):
        super(DjangoTranslator, self).__init__(**kwds)
        if context is not None:
            self.context = context
        self._django_hasher_cache = weakref.WeakKeyDictionary()
        self._passlib_hasher_cache = weakref.WeakValueDictionary()

    def reset_hashers(self):
        self._django_hasher_cache.clear()
        self._passlib_hasher_cache.clear()
        self._django_unsalted_sha1 = None

    def _get_passlib_hasher(self, passlib_name):
        """
        resolve passlib hasher by name, using context if available.
        """
        context = self.context
        if context is None:
            return registry.get_crypt_handler(passlib_name)
        else:
            return context.handler(passlib_name)

    def passlib_to_django_name(self, passlib_name):
        """
        Convert passlib hasher / name to Django hasher name.
        """
        return self.passlib_to_django(passlib_name).algorithm

    def passlib_to_django(self, passlib_hasher, cached=True):
        """
        Convert passlib hasher / name to Django hasher.

        :param passlib_hasher:
            passlib hasher / name

        :returns:
            django hasher instance
        """
        if not hasattr(passlib_hasher, 'name'):
            passlib_hasher = self._get_passlib_hasher(passlib_hasher)
        if cached:
            cache = self._django_hasher_cache
            try:
                return cache[passlib_hasher]
            except KeyError:
                pass
            result = cache[passlib_hasher] = self.passlib_to_django(passlib_hasher, cached=False)
            return result
        django_name = getattr(passlib_hasher, 'django_name', None)
        if django_name:
            return self._create_django_hasher(django_name)
        else:
            return _PasslibHasherWrapper(passlib_hasher)
    _builtin_django_hashers = dict(md5='MD5PasswordHasher')
    if DJANGO_VERSION > (2, 1):
        _builtin_django_hashers.update(bcrypt='BCryptPasswordHasher')

    def _create_django_hasher(self, django_name):
        """
        helper to create new django hasher by name.
        wraps underlying django methods.
        """
        module = sys.modules.get('passlib.ext.django.models')
        if module is None or not module.adapter.patched:
            from django.contrib.auth.hashers import get_hasher
            try:
                return get_hasher(django_name)
            except ValueError as err:
                if not str(err).startswith('Unknown password hashing algorithm'):
                    raise
        else:
            get_hashers = module.adapter._manager.getorig('django.contrib.auth.hashers:get_hashers').__wrapped__
            for hasher in get_hashers():
                if hasher.algorithm == django_name:
                    return hasher
        path = self._builtin_django_hashers.get(django_name)
        if path:
            if '.' not in path:
                path = 'django.contrib.auth.hashers.' + path
            from django.utils.module_loading import import_string
            return import_string(path)()
        raise ValueError('unknown hasher: %r' % django_name)

    def django_to_passlib_name(self, django_name):
        """
        Convert Django hasher / name to Passlib hasher name.
        """
        return self.django_to_passlib(django_name).name

    def django_to_passlib(self, django_name, cached=True):
        """
        Convert Django hasher / name to Passlib hasher / name.
        If present, CryptContext will be checked instead of main registry.

        :param django_name:
            Django hasher class or algorithm name.
            "default" allowed if context provided.

        :raises ValueError:
            if can't resolve hasher.

        :returns:
            passlib hasher or name
        """
        if hasattr(django_name, 'algorithm'):
            if isinstance(django_name, _PasslibHasherWrapper):
                return django_name.passlib_handler
            django_name = django_name.algorithm
        if cached:
            cache = self._passlib_hasher_cache
            try:
                return cache[django_name]
            except KeyError:
                pass
            result = cache[django_name] = self.django_to_passlib(django_name, cached=False)
            return result
        if django_name.startswith(PASSLIB_WRAPPER_PREFIX):
            passlib_name = django_name[len(PASSLIB_WRAPPER_PREFIX):]
            return self._get_passlib_hasher(passlib_name)
        if django_name == 'default':
            context = self.context
            if context is None:
                raise TypeError("can't determine default scheme w/ context")
            return context.handler()
        if django_name == 'unsalted_sha1':
            django_name = 'sha1'
        context = self.context
        if context is None:
            candidates = (registry.get_crypt_handler(passlib_name) for passlib_name in registry.list_crypt_handlers() if passlib_name.startswith(DJANGO_COMPAT_PREFIX) or passlib_name in _other_django_hashes)
        else:
            candidates = context.schemes(resolve=True)
        for handler in candidates:
            if getattr(handler, 'django_name', None) == django_name:
                return handler
        raise ValueError("can't translate django name to passlib name: %r" % (django_name,))

    def resolve_django_hasher(self, django_name, cached=True):
        """
        Take in a django algorithm name, return django hasher.
        """
        if hasattr(django_name, 'algorithm'):
            return django_name
        passlib_hasher = self.django_to_passlib(django_name, cached=cached)
        if django_name == 'unsalted_sha1' and passlib_hasher.name == 'django_salted_sha1':
            if not cached:
                return self._create_django_hasher(django_name)
            result = self._django_unsalted_sha1
            if result is None:
                result = self._django_unsalted_sha1 = self._create_django_hasher(django_name)
            return result
        return self.passlib_to_django(passlib_hasher, cached=cached)