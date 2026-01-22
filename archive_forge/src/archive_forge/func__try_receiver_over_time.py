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
def _try_receiver_over_time(*args, **kwargs):

    def on_error(exc, intervals, retries):
        interval = next(intervals)
        err_msg = RECEIVER_RETRY_ERROR % {'receiver': retry_fun, 'when': humanize_seconds(interval, 'in', ' ')}
        logger.error(err_msg)
        return interval
    return retry_over_time(retry_fun, Exception, args, kwargs, on_error)