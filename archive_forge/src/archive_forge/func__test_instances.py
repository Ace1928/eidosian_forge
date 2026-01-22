import os
import testtools
from unittest import mock
from troveclient.v1 import modules
def _test_instances(self, expected_query=None):
    page_mock = mock.Mock()
    self.modules._paginated = page_mock
    limit = 'test-limit'
    marker = 'test-marker'
    if not expected_query:
        expected_query = {}
    self.modules.instances(self.module_name, limit, marker, **expected_query)
    page_mock.assert_called_with('/modules/mod_1/instances', 'instances', limit, marker, query_strings=expected_query)