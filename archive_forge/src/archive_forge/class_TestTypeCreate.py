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
class TestTypeCreate(TestType):

    def setUp(self):
        super().setUp()
        self.new_volume_type = volume_fakes.create_one_volume_type(methods={'set_keys': None})
        self.project = identity_fakes.FakeProject.create_one_project()
        self.columns = ('description', 'id', 'is_public', 'name')
        self.data = (self.new_volume_type.description, self.new_volume_type.id, True, self.new_volume_type.name)
        self.volume_types_mock.create.return_value = self.new_volume_type
        self.projects_mock.get.return_value = self.project
        self.cmd = volume_type.CreateVolumeType(self.app, None)

    def test_type_create_public(self):
        arglist = ['--description', self.new_volume_type.description, '--public', self.new_volume_type.name]
        verifylist = [('description', self.new_volume_type.description), ('is_public', True), ('name', self.new_volume_type.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.create.assert_called_with(self.new_volume_type.name, description=self.new_volume_type.description, is_public=True)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_type_create_private(self):
        arglist = ['--description', self.new_volume_type.description, '--private', '--project', self.project.id, self.new_volume_type.name]
        verifylist = [('description', self.new_volume_type.description), ('is_public', False), ('project', self.project.id), ('name', self.new_volume_type.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.create.assert_called_with(self.new_volume_type.name, description=self.new_volume_type.description, is_public=False)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_type_create_with_properties(self):
        arglist = ['--property', 'myprop=myvalue', '--multiattach', '--cacheable', '--replicated', '--availability-zone', 'az1', self.new_volume_type.name]
        verifylist = [('properties', {'myprop': 'myvalue'}), ('multiattach', True), ('cacheable', True), ('replicated', True), ('availability_zones', ['az1']), ('name', self.new_volume_type.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.create.assert_called_with(self.new_volume_type.name, description=None)
        self.new_volume_type.set_keys.assert_called_once_with({'myprop': 'myvalue', 'multiattach': '<is> True', 'cacheable': '<is> True', 'replication_enabled': '<is> True', 'RESKEY:availability_zones': 'az1'})
        self.columns += ('properties',)
        self.data += (format_columns.DictColumn(None),)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_public_type_create_with_project_public(self):
        arglist = ['--project', self.project.id, self.new_volume_type.name]
        verifylist = [('is_public', None), ('project', self.project.id), ('name', self.new_volume_type.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_type_create_with_encryption(self):
        encryption_info = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': '128', 'control_location': 'front-end'}
        encryption_type = volume_fakes.create_one_encryption_volume_type(attrs=encryption_info)
        self.new_volume_type = volume_fakes.create_one_volume_type(attrs={'encryption': encryption_info})
        self.volume_types_mock.create.return_value = self.new_volume_type
        self.volume_encryption_types_mock.create.return_value = encryption_type
        encryption_columns = ('description', 'encryption', 'id', 'is_public', 'name')
        encryption_data = (self.new_volume_type.description, format_columns.DictColumn(encryption_info), self.new_volume_type.id, True, self.new_volume_type.name)
        arglist = ['--encryption-provider', 'LuksEncryptor', '--encryption-cipher', 'aes-xts-plain64', '--encryption-key-size', '128', '--encryption-control-location', 'front-end', self.new_volume_type.name]
        verifylist = [('encryption_provider', 'LuksEncryptor'), ('encryption_cipher', 'aes-xts-plain64'), ('encryption_key_size', 128), ('encryption_control_location', 'front-end'), ('name', self.new_volume_type.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.volume_types_mock.create.assert_called_with(self.new_volume_type.name, description=None)
        body = {'provider': 'LuksEncryptor', 'cipher': 'aes-xts-plain64', 'key_size': 128, 'control_location': 'front-end'}
        self.volume_encryption_types_mock.create.assert_called_with(self.new_volume_type, body)
        self.assertEqual(encryption_columns, columns)
        self.assertCountEqual(encryption_data, data)