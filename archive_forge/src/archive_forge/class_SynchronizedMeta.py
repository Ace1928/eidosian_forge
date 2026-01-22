import importlib
import sys
import threading
import time
from oslo_log import log as logging
from oslo_utils import reflection
class SynchronizedMeta(type):
    """Use an rlock to synchronize all class methods."""

    def __init__(cls, cls_name, bases, attrs):
        super(SynchronizedMeta, cls).__init__(cls_name, bases, attrs)
        rlock = threading.RLock()
        for attr_name in attrs:
            attr = getattr(cls, attr_name)
            if callable(attr):
                decorated = SynchronizedMeta._synchronize(attr, cls_name, rlock)
                setattr(cls, attr_name, decorated)

    @staticmethod
    def _synchronize(func, cls_name, rlock):

        def wrapper(*args, **kwargs):
            f_qual_name = reflection.get_callable_name(func)
            t_request = time.time()
            try:
                with rlock:
                    t_acquire = time.time()
                    LOG.debug('Method %(method_name)s acquired rlock. Waited %(time_wait)0.3fs', dict(method_name=f_qual_name, time_wait=t_acquire - t_request))
                    return func(*args, **kwargs)
            finally:
                t_release = time.time()
                LOG.debug('Method %(method_name)s released rlock. Held %(time_held)0.3fs', dict(method_name=f_qual_name, time_held=t_release - t_acquire))
        return wrapper