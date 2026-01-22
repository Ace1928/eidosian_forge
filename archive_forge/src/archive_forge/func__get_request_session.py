import os
import random
from functools import lru_cache
import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util import Retry
def _get_request_session(max_retries, backoff_factor, backoff_jitter, retry_codes, raise_on_status, respect_retry_after_header):
    """Returns a `Requests.Session` object for making an HTTP request.

    Args:
        max_retries: Maximum total number of retries.
        backoff_factor: A time factor for exponential backoff. e.g. value 5 means the HTTP
            request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
            exponential backoff.
        backoff_jitter: A random jitter to add to the backoff interval.
        retry_codes: A list of HTTP response error codes that qualifies for retry.
        raise_on_status: Whether to raise an exception, or return a response, if status falls
            in retry_codes range and retries have been exhausted.
        respect_retry_after_header: Whether to respect Retry-After header on status codes defined
            as Retry.RETRY_AFTER_STATUS_CODES or not.

    Returns:
        requests.Session object.

    """
    return _cached_get_request_session(max_retries, backoff_factor, backoff_jitter, retry_codes, raise_on_status, _pid=os.getpid(), respect_retry_after_header=respect_retry_after_header)