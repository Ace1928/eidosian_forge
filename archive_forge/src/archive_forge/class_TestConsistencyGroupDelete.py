from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
class TestConsistencyGroupDelete(TestConsistencyGroup):
    consistency_groups = volume_fakes.create_consistency_groups(count=2)

    def setUp(self):
        super().setUp()
        self.consistencygroups_mock.get = volume_fakes.get_consistency_groups(self.consistency_groups)
        self.consistencygroups_mock.delete.return_value = None
        self.cmd = consistency_group.DeleteConsistencyGroup(self.app, None)

    def test_consistency_group_delete(self):
        arglist = [self.consistency_groups[0].id]
        verifylist = [('consistency_groups', [self.consistency_groups[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.consistencygroups_mock.delete.assert_called_with(self.consistency_groups[0].id, False)
        self.assertIsNone(result)

    def test_consistency_group_delete_with_force(self):
        arglist = ['--force', self.consistency_groups[0].id]
        verifylist = [('force', True), ('consistency_groups', [self.consistency_groups[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.consistencygroups_mock.delete.assert_called_with(self.consistency_groups[0].id, True)
        self.assertIsNone(result)

    def test_delete_multiple_consistency_groups(self):
        arglist = []
        for b in self.consistency_groups:
            arglist.append(b.id)
        verifylist = [('consistency_groups', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for b in self.consistency_groups:
            calls.append(call(b.id, False))
        self.consistencygroups_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_consistency_groups_with_exception(self):
        arglist = [self.consistency_groups[0].id, 'unexist_consistency_group']
        verifylist = [('consistency_groups', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self.consistency_groups[0], exceptions.CommandError]
        with mock.patch.object(utils, 'find_resource', side_effect=find_mock_result) as find_mock:
            try:
                self.cmd.take_action(parsed_args)
                self.fail('CommandError should be raised.')
            except exceptions.CommandError as e:
                self.assertEqual('1 of 2 consistency groups failed to delete.', str(e))
            find_mock.assert_any_call(self.consistencygroups_mock, self.consistency_groups[0].id)
            find_mock.assert_any_call(self.consistencygroups_mock, 'unexist_consistency_group')
            self.assertEqual(2, find_mock.call_count)
            self.consistencygroups_mock.delete.assert_called_once_with(self.consistency_groups[0].id, False)