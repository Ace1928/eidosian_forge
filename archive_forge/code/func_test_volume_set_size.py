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
def test_volume_set_size(self):
    arglist = ['--size', '130', self._volume.display_name]
    verifylist = [('name', None), ('description', None), ('size', 130), ('property', None), ('volume', self._volume.display_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    size = 130
    self.volumes_mock.extend.assert_called_with(self._volume.id, size)
    self.assertIsNone(result)