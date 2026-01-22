import ssl
import time
import socket
import logging
from datetime import datetime, timedelta
from functools import wraps
from libcloud.utils.py3 import httplib
from libcloud.common.exceptions import RateLimitReachedError
class MinimalRetry:

    def __init__(self, retry_delay=DEFAULT_DELAY, timeout=DEFAULT_TIMEOUT, backoff=DEFAULT_BACKOFF):
        """
        Wrapper around retrying that helps to handle common transient
        exceptions.

        This minimalistic version only retries SSL errors and rate limiting.

        :param retry_delay: retry delay between the attempts.
        :param timeout: maximum time to wait.
        :param backoff: multiplier added to delay between attempts.

        :Example:

        retry_request = MinimalRetry(timeout=1, retry_delay=1, backoff=1)
        retry_request(self.connection.request)()
        """
        if retry_delay is None:
            retry_delay = DEFAULT_DELAY
        if timeout is None:
            timeout = DEFAULT_TIMEOUT
        if backoff is None:
            backoff = DEFAULT_BACKOFF
        timeout = max(timeout, 0)
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.backoff = backoff

    def __call__(self, func):

        def transform_ssl_error(function, *args, **kwargs):
            try:
                return function(*args, **kwargs)
            except ssl.SSLError as exc:
                if TRANSIENT_SSL_ERROR in str(exc):
                    raise TransientSSLError(*exc.args)
                raise exc

        @wraps(func)
        def retry_loop(*args, **kwargs):
            current_delay = self.retry_delay
            end = datetime.now() + timedelta(seconds=self.timeout)
            last_exc = None
            while datetime.now() < end:
                try:
                    return transform_ssl_error(func, *args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if isinstance(exc, RateLimitReachedError):
                        _logger.debug('You are being rate limited, backing off...')
                        retry_after = exc.retry_after if exc.retry_after else 2
                        time.sleep(retry_after)
                        current_delay = self.retry_delay
                    elif self.should_retry(exc):
                        time.sleep(current_delay)
                        current_delay *= self.backoff
                    else:
                        raise
            raise last_exc
        return retry_loop

    def should_retry(self, exception):
        return False