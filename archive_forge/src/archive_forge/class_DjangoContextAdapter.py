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
class DjangoContextAdapter(DjangoTranslator):
    """
    Object which tries to adapt a Passlib CryptContext object,
    using a Django-hasher compatible API.

    When installed in django, :mod:`!passlib.ext.django` will create
    an instance of this class, and then monkeypatch the appropriate
    methods into :mod:`!django.contrib.auth` and other appropriate places.
    """
    context = None
    _orig_make_password = None
    is_password_usable = None
    _manager = None
    enabled = True
    patched = False

    def __init__(self, context=None, get_user_category=None, **kwds):
        self.log = logging.getLogger(__name__ + '.DjangoContextAdapter')
        if context is None:
            context = CryptContext()
        super(DjangoContextAdapter, self).__init__(context=context, **kwds)
        if get_user_category:
            assert callable(get_user_category)
            self.get_user_category = get_user_category
        try:
            from functools import lru_cache
        except ImportError:
            from django.utils.lru_cache import lru_cache
        self.get_hashers = lru_cache()(self.get_hashers)
        from django.contrib.auth.hashers import make_password
        if make_password.__module__.startswith('passlib.'):
            make_password = _PatchManager.peek_unpatched_func(make_password)
        self._orig_make_password = make_password
        from django.contrib.auth.hashers import is_password_usable
        self.is_password_usable = is_password_usable
        mlog = logging.getLogger(__name__ + '.DjangoContextAdapter._manager')
        self._manager = _PatchManager(log=mlog)

    def reset_hashers(self):
        """
        Wrapper to manually reset django's hasher lookup cache
        """
        from django.contrib.auth.hashers import reset_hashers
        reset_hashers(setting='PASSWORD_HASHERS')
        super(DjangoContextAdapter, self).reset_hashers()

    def get_hashers(self):
        """
        Passlib replacement for get_hashers() --
        Return list of available django hasher classes
        """
        passlib_to_django = self.passlib_to_django
        return [passlib_to_django(hasher) for hasher in self.context.schemes(resolve=True)]

    def get_hasher(self, algorithm='default'):
        """
        Passlib replacement for get_hasher() --
        Return django hasher by name
        """
        return self.resolve_django_hasher(algorithm)

    def identify_hasher(self, encoded):
        """
        Passlib replacement for identify_hasher() --
        Identify django hasher based on hash.
        """
        handler = self.context.identify(encoded, resolve=True, required=True)
        if handler.name == 'django_salted_sha1' and encoded.startswith('sha1$$'):
            return self.get_hasher('unsalted_sha1')
        return self.passlib_to_django(handler)

    def make_password(self, password, salt=None, hasher='default'):
        """
        Passlib replacement for make_password()
        """
        if password is None:
            return self._orig_make_password(None)
        passlib_hasher = self.django_to_passlib(hasher)
        if 'salt' not in passlib_hasher.setting_kwds:
            pass
        elif hasher.startswith('unsalted_'):
            passlib_hasher = passlib_hasher.using(salt='')
        elif salt:
            passlib_hasher = passlib_hasher.using(salt=salt)
        return passlib_hasher.hash(password)

    def check_password(self, password, encoded, setter=None, preferred='default'):
        """
        Passlib replacement for check_password()
        """
        if password is None or not self.is_password_usable(encoded):
            return False
        context = self.context
        try:
            correct = context.verify(password, encoded)
        except exc.UnknownHashError:
            return False
        if not (correct and setter):
            return correct
        if preferred == 'default':
            if not context.needs_update(encoded, secret=password):
                return correct
        else:
            hasher = self.django_to_passlib(preferred)
            if hasher.identify(encoded) and (not hasher.needs_update(encoded, secret=password)):
                return correct
        setter(password)
        return correct

    def user_check_password(self, user, password):
        """
        Passlib replacement for User.check_password()
        """
        if password is None:
            return False
        hash = user.password
        if not self.is_password_usable(hash):
            return False
        cat = self.get_user_category(user)
        try:
            ok, new_hash = self.context.verify_and_update(password, hash, category=cat)
        except exc.UnknownHashError:
            return False
        if ok and new_hash is not None:
            user.password = new_hash
            user.save()
        return ok

    def user_set_password(self, user, password):
        """
        Passlib replacement for User.set_password()
        """
        if password is None:
            user.set_unusable_password()
        else:
            cat = self.get_user_category(user)
            user.password = self.context.hash(password, category=cat)

    def get_user_category(self, user):
        """
        Helper for hashing passwords per-user --
        figure out the CryptContext category for specified Django user object.
        .. note::
            This may be overridden via PASSLIB_GET_CATEGORY django setting
        """
        if user.is_superuser:
            return 'superuser'
        elif user.is_staff:
            return 'staff'
        else:
            return None
    HASHERS_PATH = 'django.contrib.auth.hashers'
    MODELS_PATH = 'django.contrib.auth.models'
    USER_CLASS_PATH = MODELS_PATH + ':User'
    FORMS_PATH = 'django.contrib.auth.forms'
    patch_locations = [(USER_CLASS_PATH + '.check_password', 'user_check_password', dict(method=True)), (USER_CLASS_PATH + '.set_password', 'user_set_password', dict(method=True)), (HASHERS_PATH + ':', 'check_password'), (HASHERS_PATH + ':', 'make_password'), (HASHERS_PATH + ':', 'get_hashers'), (HASHERS_PATH + ':', 'get_hasher'), (HASHERS_PATH + ':', 'identify_hasher'), (MODELS_PATH + ':', 'check_password'), (MODELS_PATH + ':', 'make_password'), (FORMS_PATH + ':', 'get_hasher'), (FORMS_PATH + ':', 'identify_hasher')]

    def install_patch(self):
        """
        Install monkeypatch to replace django hasher framework.
        """
        log = self.log
        if self.patched:
            log.warning('monkeypatching already applied, refusing to reapply')
            return False
        if DJANGO_VERSION < MIN_DJANGO_VERSION:
            raise RuntimeError('passlib.ext.django requires django >= %s' % (MIN_DJANGO_VERSION,))
        log.debug('preparing to monkeypatch django ...')
        manager = self._manager
        for record in self.patch_locations:
            if len(record) == 2:
                record += ({},)
            target, source, opts = record
            if target.endswith((':', ',')):
                target += source
            value = getattr(self, source)
            if opts.get('method'):
                value = _wrap_method(value)
            manager.patch(target, value)
        self.reset_hashers()
        self.patched = True
        log.debug('... finished monkeypatching django')
        return True

    def remove_patch(self):
        """
        Remove monkeypatch from django hasher framework.
        As precaution in case there are lingering refs to context,
        context object will be wiped.

        .. warning::
            This may cause problems if any other Django modules have imported
            their own copies of the patched functions, though the patched
            code has been designed to throw an error as soon as possible in
            this case.
        """
        log = self.log
        manager = self._manager
        if self.patched:
            log.debug('removing django monkeypatching...')
            manager.unpatch_all(unpatch_conflicts=True)
            self.context.load({})
            self.patched = False
            self.reset_hashers()
            log.debug('...finished removing django monkeypatching')
            return True
        if manager.isactive():
            log.warning('reverting partial monkeypatching of django...')
            manager.unpatch_all()
            self.context.load({})
            self.reset_hashers()
            log.debug('...finished removing django monkeypatching')
            return True
        log.debug('django not monkeypatched')
        return False

    def load_model(self):
        """
        Load configuration from django, and install patch.
        """
        self._load_settings()
        if self.enabled:
            try:
                self.install_patch()
            except:
                self.remove_patch()
                raise
        else:
            if self.patched:
                log.error("didn't expect monkeypatching would be applied!")
            self.remove_patch()
        log.debug('passlib.ext.django loaded')

    def _load_settings(self):
        """
        Update settings from django
        """
        from django.conf import settings
        _UNSET = object()
        config = getattr(settings, 'PASSLIB_CONFIG', _UNSET)
        if config is _UNSET:
            config = getattr(settings, 'PASSLIB_CONTEXT', _UNSET)
        if config is _UNSET:
            config = 'passlib-default'
        if config is None:
            warn("setting PASSLIB_CONFIG=None is deprecated, and support will be removed in Passlib 1.8, use PASSLIB_CONFIG='disabled' instead.", DeprecationWarning)
            config = 'disabled'
        elif not isinstance(config, (unicode, bytes, dict)):
            raise exc.ExpectedTypeError(config, 'str or dict', 'PASSLIB_CONFIG')
        get_category = getattr(settings, 'PASSLIB_GET_CATEGORY', None)
        if get_category and (not callable(get_category)):
            raise exc.ExpectedTypeError(get_category, 'callable', 'PASSLIB_GET_CATEGORY')
        if config == 'disabled':
            self.enabled = False
            return
        else:
            self.__dict__.pop('enabled', None)
        if isinstance(config, str) and '\n' not in config:
            config = get_preset_config(config)
        if get_category:
            self.get_user_category = get_category
        else:
            self.__dict__.pop('get_category', None)
        self.context.load(config)
        self.reset_hashers()