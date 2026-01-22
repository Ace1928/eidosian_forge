from datetime import datetime, timedelta, timezone
from kombu.exceptions import EncodeError
from kombu.utils.objects import cached_property
from kombu.utils.url import maybe_sanitize_url, urlparse
from celery import states
from celery.exceptions import ImproperlyConfigured
from .base import BaseBackend
def _get_database(self):
    conn = self._get_connection()
    return conn[self.database_name]