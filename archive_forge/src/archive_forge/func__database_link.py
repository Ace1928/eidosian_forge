from kombu.utils import cached_property
from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import _parse_url
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
@cached_property
def _database_link(self):
    return 'dbs/' + self._database_name