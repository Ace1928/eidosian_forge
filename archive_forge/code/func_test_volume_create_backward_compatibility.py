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
def test_volume_create_backward_compatibility(self):
    arglist = ['-c', 'display_name', '--size', str(self.new_volume.size), self.new_volume.display_name]
    verifylist = [('columns', ['display_name']), ('size', self.new_volume.size), ('name', self.new_volume.display_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.create.assert_called_with(self.new_volume.size, None, None, self.new_volume.display_name, None, None, None, None, None, None, None)
    self.assertIn('display_name', columns)
    self.assertNotIn('name', columns)
    self.assertIn(self.new_volume.display_name, data)