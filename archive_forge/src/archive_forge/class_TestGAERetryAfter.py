from test import SHORT_TIMEOUT
from test.with_dummyserver import test_connectionpool
import pytest
import dummyserver.testcase
import urllib3.exceptions
import urllib3.util.retry
import urllib3.util.url
from urllib3.contrib import appengine
@pytest.mark.usefixtures('testbed')
class TestGAERetryAfter(test_connectionpool.TestRetryAfter):

    def setup_method(self, method):
        self.manager = appengine.AppEngineManager(urlfetch_retries=False)
        self.pool = MockPool(self.host, self.port, self.manager)