from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
class TestVolumeShow(TestVolume):

    def setUp(self):
        super().setUp()
        self._volume = volume_fakes.create_one_volume()
        self.volumes_mock.get.return_value = self._volume
        self.cmd = volume.ShowVolume(self.app, None)

    def test_volume_show(self):
        arglist = [self._volume.id]
        verifylist = [('volume', self._volume.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volumes_mock.get.assert_called_with(self._volume.id)
        self.assertEqual(volume_fakes.get_volume_columns(self._volume), columns)
        self.assertCountEqual(volume_fakes.get_volume_data(self._volume), data)