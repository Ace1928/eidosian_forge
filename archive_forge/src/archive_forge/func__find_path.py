import locale
import os
from datetime import datetime
from kombu.utils.encoding import ensure_bytes
from celery import uuid
from celery.backends.base import KeyValueStoreBackend
from celery.exceptions import ImproperlyConfigured
def _find_path(self, url):
    if not url:
        raise ImproperlyConfigured(E_NO_PATH_SET)
    if url.startswith('file://localhost/'):
        return url[16:]
    if url.startswith('file://'):
        return url[7:]
    raise ImproperlyConfigured(E_PATH_NON_CONFORMING_SCHEME)