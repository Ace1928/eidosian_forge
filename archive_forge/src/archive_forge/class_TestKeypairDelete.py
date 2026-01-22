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
class TestKeypairDelete(TestKeypair):
    keypairs = compute_fakes.create_keypairs(count=2)

    def setUp(self):
        super().setUp()
        self.cmd = keypair.DeleteKeypair(self.app, None)

    def test_keypair_delete(self):
        arglist = [self.keypairs[0].name]
        verifylist = [('name', [self.keypairs[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ret = self.cmd.take_action(parsed_args)
        self.assertIsNone(ret)
        self.compute_sdk_client.delete_keypair.assert_called_with(self.keypairs[0].name, ignore_missing=False)

    def test_delete_multiple_keypairs(self):
        arglist = []
        for k in self.keypairs:
            arglist.append(k.name)
        verifylist = [('name', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for k in self.keypairs:
            calls.append(call(k.name, ignore_missing=False))
        self.compute_sdk_client.delete_keypair.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_multiple_keypairs_with_exception(self):
        arglist = [self.keypairs[0].name, 'unexist_keypair']
        verifylist = [('name', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.compute_sdk_client.delete_keypair.side_effect = [None, exceptions.CommandError]
        try:
            self.cmd.take_action(parsed_args)
            self.fail('CommandError should be raised.')
        except exceptions.CommandError as e:
            self.assertEqual('1 of 2 keys failed to delete.', str(e))
        calls = []
        for k in arglist:
            calls.append(call(k, ignore_missing=False))
        self.compute_sdk_client.delete_keypair.assert_has_calls(calls)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
    def test_keypair_delete_with_user(self, sm_mock):
        arglist = ['--user', identity_fakes.user_name, self.keypairs[0].name]
        verifylist = [('user', identity_fakes.user_name), ('name', [self.keypairs[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ret = self.cmd.take_action(parsed_args)
        self.assertIsNone(ret)
        self.compute_sdk_client.delete_keypair.assert_called_with(self.keypairs[0].name, user_id=identity_fakes.user_id, ignore_missing=False)

    @mock.patch.object(sdk_utils, 'supports_microversion', return_value=False)
    def test_keypair_delete_with_user_pre_v210(self, sm_mock):
        arglist = ['--user', identity_fakes.user_name, self.keypairs[0].name]
        verifylist = [('user', identity_fakes.user_name), ('name', [self.keypairs[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-compute-api-version 2.10 or greater is required', str(ex))