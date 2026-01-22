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
class TestTypeUnset(TestType):
    project = identity_fakes.FakeProject.create_one_project()
    volume_type = volume_fakes.create_one_volume_type(methods={'unset_keys': None})

    def setUp(self):
        super().setUp()
        self.volume_types_mock.get.return_value = self.volume_type
        self.projects_mock.get.return_value = self.project
        self.cmd = volume_type.UnsetVolumeType(self.app, None)

    def test_type_unset(self):
        arglist = ['--property', 'property', '--property', 'multi_property', self.volume_type.id]
        verifylist = [('properties', ['property', 'multi_property']), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_type.unset_keys.assert_called_once_with(['property', 'multi_property'])
        self.assertIsNone(result)

    def test_type_unset_project_access(self):
        arglist = ['--project', self.project.id, self.volume_type.id]
        verifylist = [('project', self.project.id), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.volume_type_access_mock.remove_project_access.assert_called_with(self.volume_type.id, self.project.id)

    def test_type_unset_not_called_without_project_argument(self):
        arglist = ['--project', '', self.volume_type.id]
        verifylist = [('encryption_type', False), ('project', ''), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.volume_encryption_types_mock.delete.assert_not_called()
        self.assertFalse(self.volume_type_access_mock.remove_project_access.called)

    def test_type_unset_failed_with_missing_volume_type_argument(self):
        arglist = ['--project', 'identity_fakes.project_id']
        verifylist = [('project', 'identity_fakes.project_id')]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_type_unset_encryption_type(self):
        arglist = ['--encryption-type', self.volume_type.id]
        verifylist = [('encryption_type', True), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_encryption_types_mock.delete.assert_called_with(self.volume_type)
        self.assertIsNone(result)