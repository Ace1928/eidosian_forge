import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
def test_retry_subclass_that_sets_method_whitelist(self, expect_retry_deprecation):

    class SubclassRetry(Retry):

        def __init__(self, **kwargs):
            if 'allowed_methods' in kwargs:
                raise AssertionError("This subclass likely doesn't use 'allowed_methods'")
            super(SubclassRetry, self).__init__(**kwargs)
            self.method_whitelist = self.method_whitelist | {'POST'}
    retry = SubclassRetry()
    assert retry.method_whitelist == Retry.DEFAULT_ALLOWED_METHODS | {'POST'}
    assert retry.new(read=0).method_whitelist == retry.method_whitelist
    assert retry._is_method_retryable('POST')
    assert not retry._is_method_retryable('CONNECT')
    assert retry.new(method_whitelist={'GET'}).method_whitelist == {'GET', 'POST'}
    with pytest.raises(AssertionError) as e:
        retry.new(allowed_methods={'GET'})
    assert str(e.value) == "This subclass likely doesn't use 'allowed_methods'"