import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
def _set_current_app(app):
    _tls.current_app = app