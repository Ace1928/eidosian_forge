import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
class _TLS(threading.local):
    current_app = None