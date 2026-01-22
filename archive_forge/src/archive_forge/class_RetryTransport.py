import logging
import time
from containerregistry.transport import nested
import httplib2
import six.moves.http_client
class RetryTransport(nested.NestedTransport):
    """A wrapper for the given transport which automatically retries errors."""

    def __init__(self, source_transport, max_retries=DEFAULT_MAX_RETRIES, backoff_factor=DEFAULT_BACKOFF_FACTOR, should_retry_fn=ShouldRetry):
        super(RetryTransport, self).__init__(source_transport)
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._should_retry = should_retry_fn

    def request(self, *args, **kwargs):
        """Does the request, exponentially backing off and retrying as appropriate.

    Backoff is backoff_factor * (2 ^ (retry #)) seconds.
    Args:
      *args: The sequence of positional arguments to forward to the source
        transport.
      **kwargs: The keyword arguments to forward to the source transport.

    Returns:
      The response of the HTTP request, and its contents.
    """
        retries = 0
        while True:
            try:
                return self.source_transport.request(*args, **kwargs)
            except Exception as err:
                if retries >= self._max_retries or not self._should_retry(err):
                    raise
                logging.error('Retrying after exception %s.', err)
                retries += 1
                time.sleep(self._backoff_factor * 2 ** retries)
                continue