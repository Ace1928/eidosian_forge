from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_host
class TestVolumeHost(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.host_mock = self.volume_client.services
        self.host_mock.reset_mock()