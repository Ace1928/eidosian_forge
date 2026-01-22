from datetime import datetime, timezone
from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import _parse_url
from celery import states
from celery.exceptions import ImproperlyConfigured
from .base import KeyValueStoreBackend
def _get_server(self):
    """Connect to the Elasticsearch server."""
    http_auth = None
    if self.username and self.password:
        http_auth = (self.username, self.password)
    return elasticsearch.Elasticsearch(f'{self.scheme}://{self.host}:{self.port}', retry_on_timeout=self.es_retry_on_timeout, max_retries=self.es_max_retries, timeout=self.es_timeout, http_auth=http_auth)