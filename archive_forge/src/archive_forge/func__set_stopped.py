import os
import socket
import sys
import threading
import traceback
from contextlib import contextmanager
from threading import TIMEOUT_MAX as THREAD_TIMEOUT_MAX
from celery.local import Proxy
def _set_stopped(self):
    try:
        self.__is_stopped.set()
    except TypeError:
        pass