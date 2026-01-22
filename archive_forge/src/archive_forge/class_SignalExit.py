import functools
import signal
import time
from oslo_utils import importutils
from osprofiler.drivers import base
class SignalExit(BaseException):
    pass