from collections import namedtuple
from time import sleep, time
from typing import Any, Dict
from kombu.utils.url import _parse_url as parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
def _prepare_get_request(self, key):
    """Construct the item retrieval request parameters."""
    return {'TableName': self.table_name, 'Key': {self._key_field.name: {self._key_field.data_type: key}}}