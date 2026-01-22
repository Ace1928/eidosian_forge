import sys
import threading
import warnings
import weakref
from weakref import WeakMethod
from kombu.utils.functional import retry_over_time
from celery.exceptions import CDeprecationWarning
from celery.local import PromiseProxy, Proxy
from celery.utils.functional import fun_accepts_kwargs
from celery.utils.log import get_logger
from celery.utils.time import humanize_seconds
def _boundmethod_safe_weakref(obj):
    """Get weakref constructor appropriate for `obj`.  `obj` may be a bound method.

    Bound method objects must be special-cased because they're usually garbage
    collected immediately, even if the instance they're bound to persists.

    Returns:
        a (weakref constructor, main object) tuple. `weakref constructor` is
        either :class:`weakref.ref` or :class:`weakref.WeakMethod`.  `main
        object` is the instance that `obj` is bound to if it is a bound method;
        otherwise `main object` is simply `obj.
    """
    try:
        obj.__func__
        obj.__self__
        return (WeakMethod, obj.__self__)
    except AttributeError:
        return (weakref.ref, obj)