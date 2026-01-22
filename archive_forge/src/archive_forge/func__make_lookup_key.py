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
def _make_lookup_key(receiver, sender, dispatch_uid):
    if dispatch_uid:
        return (dispatch_uid, _make_id(sender))
    else:
        return (_make_id(receiver), _make_id(sender))