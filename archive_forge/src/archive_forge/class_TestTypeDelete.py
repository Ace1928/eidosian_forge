from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_type
class TestTypeDelete(TestType):
    volume_types = volume_fakes.create_volume_types(count=2)

    def setUp(self):
        super().setUp()
        self.volume_types_mock.get = volume_fakes.get_volume_types(self.volume_types)
        self.volume_types_mock.delete.return_value = None
        self.cmd = volume_type.DeleteVolumeType(self.app, None)

    def test_type_delete(self):
        arglist = [self.volume_types[0].id]
        verifylist = [('volume_types', [self.volume_types[0].id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_types_mock.delete.assert_called_with(self.volume_types[0])
        self.assertIsNone(result)

    def test_delete_multiple_types(self):
        arglist = []
        for t in self.volume_types:
            arglist.append(t.id)
        verifylist = [('volume_types', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for t in self.volume_types:
            calls.append(call(t))
        self.volume_types_mock.delete.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_types_with_exception(self):
        arglist = [self.volume_types[0].id, 'unexist_type']
        verifylist = [('volume_types', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        find_mock_result = [self.volume_types[0], exceptions.CommandError]
        with mock.patch.object(utils, 'find_resource', side_effect=find_mock_result) as find_mock:
            try:
                self.cmd.take_action(parsed_args)
                self.fail('CommandError should be raised.')
            except exceptions.CommandError as e:
                self.assertEqual('1 of 2 volume types failed to delete.', str(e))
            find_mock.assert_any_call(self.volume_types_mock, self.volume_types[0].id)
            find_mock.assert_any_call(self.volume_types_mock, 'unexist_type')
            self.assertEqual(2, find_mock.call_count)
            self.volume_types_mock.delete.assert_called_once_with(self.volume_types[0])