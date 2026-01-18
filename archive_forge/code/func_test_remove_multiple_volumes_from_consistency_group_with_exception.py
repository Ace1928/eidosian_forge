from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
@mock.patch.object(consistency_group.LOG, 'error')
def test_remove_multiple_volumes_from_consistency_group_with_exception(self, mock_error):
    volume = volume_fakes.create_one_volume()
    arglist = [self._consistency_group.id, volume.id, 'unexist_volume']
    verifylist = [('consistency_group', self._consistency_group.id), ('volumes', [volume.id, 'unexist_volume'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [volume, exceptions.CommandError, self._consistency_group]
    with mock.patch.object(utils, 'find_resource', side_effect=find_mock_result) as find_mock:
        result = self.cmd.take_action(parsed_args)
        mock_error.assert_called_with('1 of 2 volumes failed to remove.')
        self.assertIsNone(result)
        find_mock.assert_any_call(self.consistencygroups_mock, self._consistency_group.id)
        find_mock.assert_any_call(self.volumes_mock, volume.id)
        find_mock.assert_any_call(self.volumes_mock, 'unexist_volume')
        self.assertEqual(3, find_mock.call_count)
        self.consistencygroups_mock.update.assert_called_once_with(self._consistency_group.id, remove_volumes=volume.id)