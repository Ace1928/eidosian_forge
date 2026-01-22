import uuid
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cleanup
class TestBlockStorage(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.worker_mock = self.volume_client.workers
        self.worker_mock.reset_mock()