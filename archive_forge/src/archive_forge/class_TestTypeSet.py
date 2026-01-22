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
class TestTypeSet(TestType):

    def setUp(self):
        super().setUp()
        self.project = identity_fakes.FakeProject.create_one_project()
        self.projects_mock.get.return_value = self.project
        self.volume_type = volume_fakes.create_one_volume_type(methods={'set_keys': None})
        self.volume_types_mock.get.return_value = self.volume_type
        self.volume_encryption_types_mock.create.return_value = None
        self.volume_encryption_types_mock.update.return_value = None
        self.cmd = volume_type.SetVolumeType(self.app, None)

    def test_type_set(self):
        arglist = ['--name', 'new_name', '--description', 'new_description', '--private', self.volume_type.id]
        verifylist = [('name', 'new_name'), ('description', 'new_description'), ('properties', None), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'name': 'new_name', 'description': 'new_description', 'is_public': False}
        self.volume_types_mock.update.assert_called_with(self.volume_type.id, **kwargs)
        self.assertIsNone(result)
        self.volume_type_access_mock.add_project_access.assert_not_called()
        self.volume_encryption_types_mock.update.assert_not_called()
        self.volume_encryption_types_mock.create.assert_not_called()

    def test_type_set_property(self):
        arglist = ['--property', 'myprop=myvalue', '--multiattach', '--cacheable', '--replicated', '--availability-zone', 'az1', self.volume_type.id]
        verifylist = [('name', None), ('description', None), ('properties', {'myprop': 'myvalue'}), ('multiattach', True), ('cacheable', True), ('replicated', True), ('availability_zones', ['az1']), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.volume_type.set_keys.assert_called_once_with({'myprop': 'myvalue', 'multiattach': '<is> True', 'cacheable': '<is> True', 'replication_enabled': '<is> True', 'RESKEY:availability_zones': 'az1'})
        self.volume_type_access_mock.add_project_access.assert_not_called()
        self.volume_encryption_types_mock.update.assert_not_called()
        self.volume_encryption_types_mock.create.assert_not_called()

    def test_type_set_with_empty_project(self):
        arglist = ['--project', '', self.volume_type.id]
        verifylist = [('project', ''), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.volume_type.set_keys.assert_not_called()
        self.volume_type_access_mock.add_project_access.assert_not_called()
        self.volume_encryption_types_mock.update.assert_not_called()
        self.volume_encryption_types_mock.create.assert_not_called()

    def test_type_set_with_project(self):
        arglist = ['--project', self.project.id, self.volume_type.id]
        verifylist = [('project', self.project.id), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.volume_type.set_keys.assert_not_called()
        self.volume_type_access_mock.add_project_access.assert_called_with(self.volume_type.id, self.project.id)
        self.volume_encryption_types_mock.update.assert_not_called()
        self.volume_encryption_types_mock.create.assert_not_called()

    def test_type_set_with_new_encryption(self):
        self.volume_encryption_types_mock.update.side_effect = exceptions.NotFound('NotFound')
        arglist = ['--encryption-provider', 'LuksEncryptor', '--encryption-cipher', 'aes-xts-plain64', '--encryption-key-size', '128', '--encryption-control-location', 'front-end', self.volume_type.id]
        verifylist = [('encryption_provider', 'LuksEncryptor'), ('encryption_cipher', 'aes-xts-plain64'), ('encryption_key_size', 128), ('encryption_control_location', 'front-end'), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        body = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
        self.volume_encryption_types_mock.update.assert_called_with(self.volume_type, body)
        self.volume_encryption_types_mock.create.assert_called_with(self.volume_type, body)

    @mock.patch.object(utils, 'find_resource')
    def test_type_set_with_existing_encryption(self, mock_find):
        mock_find.side_effect = [self.volume_type, 'existing_encryption_type']
        arglist = ['--encryption-provider', 'LuksEncryptor', '--encryption-cipher', 'aes-xts-plain64', '--encryption-control-location', 'front-end', self.volume_type.id]
        verifylist = [('encryption_provider', 'LuksEncryptor'), ('encryption_cipher', 'aes-xts-plain64'), ('encryption_control_location', 'front-end'), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertIsNone(result)
        self.volume_type.set_keys.assert_not_called()
        self.volume_type_access_mock.add_project_access.assert_not_called()
        body = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'control_location': 'front-end'}
        self.volume_encryption_types_mock.update.assert_called_with(self.volume_type, body)
        self.volume_encryption_types_mock.create.assert_not_called()

    def test_type_set_new_encryption_without_provider(self):
        self.volume_encryption_types_mock.update.side_effect = exceptions.NotFound('NotFound')
        arglist = ['--encryption-cipher', 'aes-xts-plain64', '--encryption-key-size', '128', '--encryption-control-location', 'front-end', self.volume_type.id]
        verifylist = [('encryption_cipher', 'aes-xts-plain64'), ('encryption_key_size', 128), ('encryption_control_location', 'front-end'), ('volume_type', self.volume_type.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertEqual('Command Failed: One or more of the operations failed', str(exc))
        self.volume_type.set_keys.assert_not_called()
        self.volume_type_access_mock.add_project_access.assert_not_called()
        body = {'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
        self.volume_encryption_types_mock.update.assert_called_with(self.volume_type, body)
        self.volume_encryption_types_mock.create.assert_not_called()