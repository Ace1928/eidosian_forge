from unittest import mock
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import volumes
def test_volume_update_server_volume(self):
    v = self.cs.volumes.update_server_volume(server_id=1234, src_volid='Work', dest_volid='Work', delete_on_termination=True)
    self.assert_request_id(v, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('PUT', '/servers/1234/os-volume_attachments/Work')
    self.assertIsInstance(v, volumes.Volume)