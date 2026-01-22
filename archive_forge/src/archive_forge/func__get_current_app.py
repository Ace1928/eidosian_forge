import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
def _get_current_app():
    if default_app is None:
        from celery.app.base import Celery
        set_default_app(Celery('default', fixups=[], set_as_current=False, loader=os.environ.get('CELERY_LOADER') or 'default'))
    return _tls.current_app or default_app