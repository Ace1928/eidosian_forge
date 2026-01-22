from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from os import getpid
from threading import RLock
from kombu.utils.encoding import bytes_to_str
from kombu.utils.functional import dictfilter
from kombu.utils.url import url_to_parts
from celery.exceptions import ImproperlyConfigured
from .base import KeyValueStoreBackend
def _is_bucket_lifecycle_rule_exists(self):
    bucket = self.bucket
    bucket.reload()
    for rule in bucket.lifecycle_rules:
        if rule['action']['type'] == 'Delete':
            return True
    return False