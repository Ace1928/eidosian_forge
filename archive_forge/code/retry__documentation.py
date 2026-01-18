import logging
import time
from containerregistry.transport import nested
import httplib2
import six.moves.http_client
Does the request, exponentially backing off and retrying as appropriate.

    Backoff is backoff_factor * (2 ^ (retry #)) seconds.
    Args:
      *args: The sequence of positional arguments to forward to the source
        transport.
      **kwargs: The keyword arguments to forward to the source transport.

    Returns:
      The response of the HTTP request, and its contents.
    