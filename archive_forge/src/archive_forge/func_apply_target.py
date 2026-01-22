import sys
from time import monotonic
from greenlet import GreenletExit
from kombu.asynchronous import timer as _timer
from celery import signals
from . import base
def apply_target(target, args=(), kwargs=None, callback=None, accept_callback=None, getpid=None):
    kwargs = {} if not kwargs else kwargs
    return base.apply_target(target, args, kwargs, callback, accept_callback, pid=getpid())