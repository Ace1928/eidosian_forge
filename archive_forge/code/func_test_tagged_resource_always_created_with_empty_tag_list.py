from unittest import mock
from keystoneauth1 import adapter
from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_tagged_resource_always_created_with_empty_tag_list(self):
    res = self.sot
    self.assertIsNotNone(res.tags)
    self.assertEqual(res.tags, list())