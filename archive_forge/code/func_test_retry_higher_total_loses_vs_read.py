import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_higher_total_loses_vs_read(self):
    """A lower read timeout than the total is honored"""
    error = ReadTimeoutError(None, '/', 'read timed out')
    retry = Retry(read=2, total=3)
    retry = retry.increment(method='GET', error=error)
    retry = retry.increment(method='GET', error=error)
    with pytest.raises(MaxRetryError):
        retry.increment(method='GET', error=error)