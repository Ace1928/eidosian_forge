import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
@pytest.mark.parametrize('respect_retry_after_header', [True, False])
def test_respect_retry_after_header_propagated(self, respect_retry_after_header):
    retry = Retry(respect_retry_after_header=respect_retry_after_header)
    new_retry = retry.new()
    assert new_retry.respect_retry_after_header == respect_retry_after_header