from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.image.v1 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v1 import fakes as volume_fakes
from openstackclient.volume.v1 import volume
def test_volume_show_backward_compatibility(self):
    arglist = ['-c', 'display_name', self._volume.id]
    verifylist = [('columns', ['display_name']), ('volume', self._volume.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.get.assert_called_with(self._volume.id)
    self.assertIn('display_name', columns)
    self.assertNotIn('name', columns)
    self.assertIn(self._volume.display_name, data)