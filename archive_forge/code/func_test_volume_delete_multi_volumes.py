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
def test_volume_delete_multi_volumes(self):
    volumes = self.setup_volumes_mock(count=3)
    arglist = [v.id for v in volumes]
    verifylist = [('force', False), ('purge', False), ('volumes', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    calls = [call(v.id, cascade=False) for v in volumes]
    self.volumes_mock.delete.assert_has_calls(calls)
    self.assertIsNone(result)