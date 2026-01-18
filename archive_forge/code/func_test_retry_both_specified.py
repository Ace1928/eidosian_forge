import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_both_specified(self):
    """Total can win if it's lower than the connect value"""
    error = ConnectTimeoutError()
    retry = Retry(connect=3, total=2)
    retry = retry.increment(error=error)
    retry = retry.increment(error=error)
    with pytest.raises(MaxRetryError) as e:
        retry.increment(error=error)
    assert e.value.reason == error