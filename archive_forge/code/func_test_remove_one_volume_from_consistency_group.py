from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_remove_one_volume_from_consistency_group(self):
    volume = volume_fakes.create_one_volume()
    self.volumes_mock.get.return_value = volume
    arglist = [self._consistency_group.id, volume.id]
    verifylist = [('consistency_group', self._consistency_group.id), ('volumes', [volume.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'remove_volumes': volume.id}
    self.consistencygroups_mock.update.assert_called_once_with(self._consistency_group.id, **kwargs)
    self.assertIsNone(result)