from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
class TestVolumeSnapshot(volume_fakes.TestVolume):

    def setUp(self):
        super().setUp()
        self.snapshots_mock = self.volume_client.volume_snapshots
        self.snapshots_mock.reset_mock()
        self.volumes_mock = self.volume_client.volumes
        self.volumes_mock.reset_mock()
        self.project_mock = self.app.client_manager.identity.projects
        self.project_mock.reset_mock()