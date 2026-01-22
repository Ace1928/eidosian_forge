import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _retry_if_failed(partial, attempts=_UPLOAD_ATTEMPTS, sleep_seconds=_SLEEP_SECONDS, exceptions=None):
    if exceptions is None:
        exceptions = (botocore.exceptions.EndpointConnectionError,)
    for attempt in range(attempts):
        try:
            return partial()
        except exceptions:
            logger.critical('Unable to connect to the endpoint. Check your network connection. Sleeping and retrying %d more times before giving up.' % (attempts - attempt - 1))
            time.sleep(sleep_seconds)
    else:
        logger.critical('Unable to connect to the endpoint. Giving up.')
        raise IOError('Unable to connect to the endpoint after %d attempts' % attempts)