import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
class SubclassRetry(Retry):

    def __init__(self, **kwargs):
        if 'allowed_methods' in kwargs:
            raise AssertionError("This subclass likely doesn't use 'allowed_methods'")
        super(SubclassRetry, self).__init__(**kwargs)
        self.method_whitelist = self.method_whitelist | {'POST'}