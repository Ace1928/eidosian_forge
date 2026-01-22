import copy
from unittest import mock
from unittest.mock import call
import uuid
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import keypair
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestKeypairList(TestKeypair):
    keypairs = compute_fakes.create_keypairs(count=1)

    def setUp(self):
        super().setUp()
        self.compute_sdk_client.keypairs.return_value = self.keypairs
        self.cmd = keypair.ListKeypair(self.app, None)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_keypair_list_no_options(self, sm_mock):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.keypairs.assert_called_with()
        self.assertEqual(('Name', 'Fingerprint'), columns)
        self.assertEqual(((self.keypairs[0].name, self.keypairs[0].fingerprint),), tuple(data))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_keypair_list_v22(self, sm_mock):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.keypairs.assert_called_with()
        self.assertEqual(('Name', 'Fingerprint', 'Type'), columns)
        self.assertEqual(((self.keypairs[0].name, self.keypairs[0].fingerprint, self.keypairs[0].type),), tuple(data))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_keypair_list_with_user(self, sm_mock):
        users_mock = self.app.client_manager.identity.users
        users_mock.reset_mock()
        users_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.USER), loaded=True)
        arglist = ['--user', identity_fakes.user_name]
        verifylist = [('user', identity_fakes.user_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        users_mock.get.assert_called_with(identity_fakes.user_name)
        self.compute_sdk_client.keypairs.assert_called_with(user_id=identity_fakes.user_id)
        self.assertEqual(('Name', 'Fingerprint', 'Type'), columns)
        self.assertEqual(((self.keypairs[0].name, self.keypairs[0].fingerprint, self.keypairs[0].type),), tuple(data))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_keypair_list_with_user_pre_v210(self, sm_mock):
        arglist = ['--user', identity_fakes.user_name]
        verifylist = [('user', identity_fakes.user_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.10 or greater is required', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_keypair_list_with_project(self, sm_mock):
        projects_mock = self.app.client_manager.identity.tenants
        projects_mock.reset_mock()
        projects_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.PROJECT), loaded=True)
        users_mock = self.app.client_manager.identity.users
        users_mock.reset_mock()
        users_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.USER), loaded=True)]
        arglist = ['--project', identity_fakes.project_name]
        verifylist = [('project', identity_fakes.project_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        projects_mock.get.assert_called_with(identity_fakes.project_name)
        users_mock.list.assert_called_with(tenant_id=identity_fakes.project_id)
        self.compute_sdk_client.keypairs.assert_called_with(user_id=identity_fakes.user_id)
        self.assertEqual(('Name', 'Fingerprint', 'Type'), columns)
        self.assertEqual(((self.keypairs[0].name, self.keypairs[0].fingerprint, self.keypairs[0].type),), tuple(data))

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_keypair_list_with_project_pre_v210(self, sm_mock):
        arglist = ['--project', identity_fakes.project_name]
        verifylist = [('project', identity_fakes.project_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.10 or greater is required', str(ex))

    def test_keypair_list_conflicting_user_options(self):
        arglist = ['--user', identity_fakes.user_name, '--project', identity_fakes.project_name]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, None)

    @mock.patch.object(sdk_utils, 'supports_microversion', new=mock.Mock(return_value=True))
    def test_keypair_list_with_limit(self):
        arglist = ['--limit', '1']
        verifylist = [('limit', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.keypairs.assert_called_with(limit=1)

    @mock.patch.object(sdk_utils, 'supports_microversion', new=mock.Mock(return_value=False))
    def test_keypair_list_with_limit_pre_v235(self):
        arglist = ['--limit', '1']
        verifylist = [('limit', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.35 or greater is required', str(ex))

    @mock.patch.object(sdk_utils, 'supports_microversion', new=mock.Mock(return_value=True))
    def test_keypair_list_with_marker(self):
        arglist = ['--marker', 'test_kp']
        verifylist = [('marker', 'test_kp')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.compute_sdk_client.keypairs.assert_called_with(marker='test_kp')

    @mock.patch.object(sdk_utils, 'supports_microversion', new=mock.Mock(return_value=False))
    def test_keypair_list_with_marker_pre_v235(self):
        arglist = ['--marker', 'test_kp']
        verifylist = [('marker', 'test_kp')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.35 or greater is required', str(ex))