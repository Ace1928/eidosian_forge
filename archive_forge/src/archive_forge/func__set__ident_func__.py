import os
import socket
import sys
import threading
import traceback
from contextlib import contextmanager
from threading import TIMEOUT_MAX as THREAD_TIMEOUT_MAX
from celery.local import Proxy
def _set__ident_func__(self, value):
    object.__setattr__(self._local, '__ident_func__', value)