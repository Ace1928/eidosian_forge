from unittest import mock
from openstack.tests.unit import base
def verify_wait_for_status(self, test_method, mock_method='openstack.resource.wait_for_status', **kwargs):
    self._verify(mock_method, test_method, **kwargs)