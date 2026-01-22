from kombu.utils import cached_property
from kombu.utils.encoding import bytes_to_str
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
from .base import KeyValueStoreBackend
@cached_property
def _blob_service_client(self):
    """Return the Azure Storage Blob service client.

        If this is the first call to the property, the client is created and
        the container is created if it doesn't yet exist.

        """
    client = BlobServiceClient.from_connection_string(self._connection_string, connection_timeout=self._connection_timeout, read_timeout=self._read_timeout)
    try:
        client.create_container(name=self._container_name)
        msg = f'Container created with name {self._container_name}.'
    except ResourceExistsError:
        msg = f'Container with name {self._container_name} already.exists. This will not be created.'
    LOGGER.info(msg)
    return client