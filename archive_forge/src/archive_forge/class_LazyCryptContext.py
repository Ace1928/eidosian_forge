from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
class LazyCryptContext(CryptContext):
    """CryptContext subclass which doesn't load handlers until needed.

    This is a subclass of CryptContext which takes in a set of arguments
    exactly like CryptContext, but won't import any handlers
    (or even parse its arguments) until
    the first time one of its methods is accessed.

    :arg schemes:
        The first positional argument can be a list of schemes, or omitted,
        just like CryptContext.

    :param onload:

        If a callable is passed in via this keyword,
        it will be invoked at lazy-load time
        with the following signature:
        ``onload(**kwds) -> kwds``;
        where ``kwds`` is all the additional kwds passed to LazyCryptContext.
        It should perform any additional deferred initialization,
        and return the final dict of options to be passed to CryptContext.

        .. versionadded:: 1.6

    :param create_policy:

        .. deprecated:: 1.6
            This option will be removed in Passlib 1.8,
            applications should use ``onload`` instead.

    :param kwds:

        All additional keywords are passed to CryptContext;
        or to the *onload* function (if provided).

    This is mainly used internally by modules such as :mod:`passlib.apps`,
    which define a large number of contexts, but only a few of them will be needed
    at any one time. Use of this class saves the memory needed to import
    the specified handlers until the context instance is actually accessed.
    As well, it allows constructing a context at *module-init* time,
    but using :func:`!onload()` to provide dynamic configuration
    at *application-run* time.

    .. note:: 
        This class is only useful if you're referencing handler objects by name,
        and don't want them imported until runtime. If you want to have the config
        validated before your application runs, or are passing in already-imported
        handler instances, you should use :class:`CryptContext` instead.

    .. versionadded:: 1.4
    """
    _lazy_kwds = None

    def __init__(self, schemes=None, **kwds):
        if schemes is not None:
            kwds['schemes'] = schemes
        self._lazy_kwds = kwds

    def _lazy_init(self):
        kwds = self._lazy_kwds
        if 'create_policy' in kwds:
            warn("The CryptPolicy class, and LazyCryptContext's ``create_policy`` keyword have been deprecated as of Passlib 1.6, and will be removed in Passlib 1.8; please use the ``onload`` keyword instead.", DeprecationWarning)
            create_policy = kwds.pop('create_policy')
            result = create_policy(**kwds)
            policy = CryptPolicy.from_source(result, _warn=False)
            kwds = policy._context.to_dict()
        elif 'onload' in kwds:
            onload = kwds.pop('onload')
            kwds = onload(**kwds)
        del self._lazy_kwds
        super(LazyCryptContext, self).__init__(**kwds)
        self.__class__ = CryptContext

    def __getattribute__(self, attr):
        if (not attr.startswith('_') or attr.startswith('__')) and self._lazy_kwds is not None:
            self._lazy_init()
        return object.__getattribute__(self, attr)