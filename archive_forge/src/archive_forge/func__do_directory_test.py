import locale
import os
from datetime import datetime
from kombu.utils.encoding import ensure_bytes
from celery import uuid
from celery.backends.base import KeyValueStoreBackend
from celery.exceptions import ImproperlyConfigured
def _do_directory_test(self, key):
    try:
        self.set(key, b'test value')
        assert self.get(key) == b'test value'
        self.delete(key)
    except OSError:
        raise ImproperlyConfigured(E_PATH_INVALID)