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
def _connect_proxy(self, fun, sender, weak, dispatch_uid):
    return self.connect(fun, sender=sender._get_current_object(), weak=weak, dispatch_uid=dispatch_uid)