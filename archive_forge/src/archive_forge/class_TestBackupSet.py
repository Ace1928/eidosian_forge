from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_backup
class TestBackupSet(TestBackupLegacy):
    backup = volume_fakes.create_one_backup(attrs={'metadata': {'wow': 'cool'}})

    def setUp(self):
        super().setUp()
        self.backups_mock.get.return_value = self.backup
        self.cmd = volume_backup.SetVolumeBackup(self.app, None)

    def test_backup_set_name(self):
        self.volume_client.api_version = api_versions.APIVersion('3.9')
        arglist = ['--name', 'new_name', self.backup.id]
        verifylist = [('name', 'new_name'), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.backups_mock.update.assert_called_once_with(self.backup.id, **{'name': 'new_name'})
        self.assertIsNone(result)

    def test_backup_set_name_pre_v39(self):
        self.volume_client.api_version = api_versions.APIVersion('3.8')
        arglist = ['--name', 'new_name', self.backup.id]
        verifylist = [('name', 'new_name'), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.9 or greater', str(exc))

    def test_backup_set_description(self):
        self.volume_client.api_version = api_versions.APIVersion('3.9')
        arglist = ['--description', 'new_description', self.backup.id]
        verifylist = [('name', None), ('description', 'new_description'), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'description': 'new_description'}
        self.backups_mock.update.assert_called_once_with(self.backup.id, **kwargs)
        self.assertIsNone(result)

    def test_backup_set_description_pre_v39(self):
        self.volume_client.api_version = api_versions.APIVersion('3.8')
        arglist = ['--description', 'new_description', self.backup.id]
        verifylist = [('name', None), ('description', 'new_description'), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.9 or greater', str(exc))

    def test_backup_set_state(self):
        arglist = ['--state', 'error', self.backup.id]
        verifylist = [('state', 'error'), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.backups_mock.reset_state.assert_called_once_with(self.backup.id, 'error')
        self.assertIsNone(result)

    def test_backup_set_state_failed(self):
        self.backups_mock.reset_state.side_effect = exceptions.CommandError()
        arglist = ['--state', 'error', self.backup.id]
        verifylist = [('state', 'error'), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('One or more of the set operations failed', str(e))
        self.backups_mock.reset_state.assert_called_with(self.backup.id, 'error')

    def test_backup_set_no_property(self):
        self.volume_client.api_version = api_versions.APIVersion('3.43')
        arglist = ['--no-property', self.backup.id]
        verifylist = [('no_property', True), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'metadata': {}}
        self.backups_mock.update.assert_called_once_with(self.backup.id, **kwargs)
        self.assertIsNone(result)

    def test_backup_set_no_property_pre_v343(self):
        self.volume_client.api_version = api_versions.APIVersion('3.42')
        arglist = ['--no-property', self.backup.id]
        verifylist = [('no_property', True), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.43 or greater', str(exc))

    def test_backup_set_property(self):
        self.volume_client.api_version = api_versions.APIVersion('3.43')
        arglist = ['--property', 'foo=bar', self.backup.id]
        verifylist = [('properties', {'foo': 'bar'}), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        kwargs = {'metadata': {'wow': 'cool', 'foo': 'bar'}}
        self.backups_mock.update.assert_called_once_with(self.backup.id, **kwargs)
        self.assertIsNone(result)

    def test_backup_set_property_pre_v343(self):
        self.volume_client.api_version = api_versions.APIVersion('3.42')
        arglist = ['--property', 'foo=bar', self.backup.id]
        verifylist = [('properties', {'foo': 'bar'}), ('backup', self.backup.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.43 or greater', str(exc))