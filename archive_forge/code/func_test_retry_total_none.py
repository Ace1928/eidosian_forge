import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_total_none(self):
    """if Total is none, connect error should take precedence"""
    error = ConnectTimeoutError()
    retry = Retry(connect=2, total=None)
    retry = retry.increment(error=error)
    retry = retry.increment(error=error)
    with pytest.raises(MaxRetryError) as e:
        retry.increment(error=error)
    assert e.value.reason == error
    error = ReadTimeoutError(None, '/', 'read timed out')
    retry = Retry(connect=2, total=None)
    retry = retry.increment(method='GET', error=error)
    retry = retry.increment(method='GET', error=error)
    retry = retry.increment(method='GET', error=error)
    assert not retry.is_exhausted()