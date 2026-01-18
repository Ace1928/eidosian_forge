from unittest import mock
from openstack import exceptions
from openstack import proxy
from openstack import resource
from openstack.tests.unit import base
def test_raises_unknown_service(self):
    self.assertRaises(exceptions.SDKException, self.cloud.search_resources, 'wrong_service.wrong_resource', 'name')