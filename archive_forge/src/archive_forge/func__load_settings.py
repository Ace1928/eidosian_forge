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