from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
class VolumesV249Test(VolumesTest):
    api_version = '2.49'

    def test_create_server_volume_with_tag(self):
        v = self.cs.volumes.create_server_volume(server_id=1234, volume_id='15e59938-07d5-11e1-90e3-e3dffe0c5983', device='/dev/vdb', tag='test_tag')
        self.assert_request_id(v, fakes.FAKE_REQUEST_ID_LIST)
        self.cs.assert_called('POST', '/servers/1234/os-volume_attachments', {'volumeAttachment': {'volumeId': '15e59938-07d5-11e1-90e3-e3dffe0c5983', 'device': '/dev/vdb', 'tag': 'test_tag'}})
        self.assertIsInstance(v, volumes.Volume)

    def test_delete_server_volume_with_exception(self):
        self.assertRaises(TypeError, self.cs.volumes.delete_server_volume, '1234')
        self.assertRaises(TypeError, self.cs.volumes.delete_server_volume, '1234', volume_id='Work', attachment_id='123')

    @mock.patch('warnings.warn')
    def test_delete_server_volume_with_warn(self, mock_warn):
        self.cs.volumes.delete_server_volume(1234, volume_id=None, attachment_id='Work')
        mock_warn.assert_called_once()