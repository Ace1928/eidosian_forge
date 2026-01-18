import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
@pytest.mark.parametrize('value, expected', [('0', 0), ('1000', 1000), ('\t42 ', 42)])
def test_parse_retry_after(self, value, expected):
    retry = Retry()
    assert retry.parse_retry_after(value) == expected