from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
@mock.patch.object(meta, 'get_server_external_ipv4')
def test_private_interface_ip(self, mock_get_server_external_ipv4):
    mock_get_server_external_ipv4.return_value = PUBLIC_V4
    cloud = FakeCloud()
    cloud.private = True
    hostvars = meta.get_hostvars_from_server(cloud, meta.obj_to_munch(standard_fake_server))
    self.assertEqual(PRIVATE_V4, hostvars['interface_ip'])