import copy
from unittest import mock
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import session as ksa_session
from openstack.config import cloud_region
from openstack.config import defaults
from openstack import exceptions
from openstack.tests.unit.config import base
from openstack import version as openstack_version
def test_rackspace_workaround_only_rax(self):
    cc = cloud_region.CloudRegion('test1', 'DFW', {'region_name': 'DFW', 'auth': {'project_id': '123456'}, 'block_storage_endpoint_override': 'https://example.com/v2/'})
    self.assertEqual('https://example.com/v2/', cc.get_endpoint('block-storage'))