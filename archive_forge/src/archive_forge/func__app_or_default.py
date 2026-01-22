import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
def _app_or_default(app=None):
    if app is None:
        return get_current_app()
    return app