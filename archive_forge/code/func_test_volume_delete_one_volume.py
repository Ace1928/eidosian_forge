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
def test_volume_delete_one_volume(self):
    volumes = self.setup_volumes_mock(count=1)
    arglist = [volumes[0].id]
    verifylist = [('force', False), ('purge', False), ('volumes', [volumes[0].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.volumes_mock.delete.assert_called_once_with(volumes[0].id, cascade=False)
    self.assertIsNone(result)