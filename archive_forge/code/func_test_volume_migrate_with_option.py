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
def test_volume_migrate_with_option(self):
    arglist = ['--force-host-copy', '--lock-volume', '--host', 'host@backend-name#pool', self._volume.id]
    verifylist = [('force_host_copy', True), ('lock_volume', True), ('host', 'host@backend-name#pool'), ('volume', self._volume.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.volumes_mock.get.assert_called_once_with(self._volume.id)
    self.volumes_mock.migrate_volume.assert_called_once_with(self._volume.id, 'host@backend-name#pool', True, True)
    self.assertIsNone(result)