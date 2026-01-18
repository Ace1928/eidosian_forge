import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_read_zero(self):
    """No second chances on read timeouts, by default"""
    error = ReadTimeoutError(None, '/', 'read timed out')
    retry = Retry(read=0)
    with pytest.raises(MaxRetryError) as e:
        retry.increment(method='GET', error=error)
    assert e.value.reason == error