from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
def test_unknown_volume_exception(self):
    mock_cloud = mock.MagicMock()

    class FakeException(Exception):
        pass

    def side_effect(*args):
        raise FakeException('No Volumes')
    mock_cloud.get_volumes.side_effect = side_effect
    self.assertRaises(FakeException, meta.get_hostvars_from_server, mock_cloud, meta.obj_to_munch(standard_fake_server))