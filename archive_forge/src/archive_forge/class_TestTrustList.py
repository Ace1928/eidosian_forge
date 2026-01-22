import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import trust
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestTrustList(TestTrust):

    def setUp(self):
        super(TestTrustList, self).setUp()
        self.trusts_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.TRUST), loaded=True)]
        self.cmd = trust.ListTrust(self.app, None)

    def test_trust_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.trusts_mock.list.assert_called_with(trustor_user=None, trustee_user=None)
        collist = ('ID', 'Expires At', 'Impersonation', 'Project ID', 'Trustee User ID', 'Trustor User ID')
        self.assertEqual(collist, columns)
        datalist = ((identity_fakes.trust_id, identity_fakes.trust_expires, identity_fakes.trust_impersonation, identity_fakes.project_id, identity_fakes.user_id, identity_fakes.user_id),)
        self.assertEqual(datalist, tuple(data))

    def test_trust_list_auth_user(self):
        auth_ref = self.app.client_manager.auth_ref = mock.Mock()
        auth_ref.user_id.return_value = identity_fakes.user_id
        arglist = ['--auth-user']
        verifylist = [('trustor', None), ('trustee', None), ('authuser', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.trusts_mock.list.assert_any_call(trustor_user=self.users_mock.get())
        self.trusts_mock.list.assert_any_call(trustee_user=self.users_mock.get())
        collist = ('ID', 'Expires At', 'Impersonation', 'Project ID', 'Trustee User ID', 'Trustor User ID')
        self.assertEqual(collist, columns)
        datalist = ((identity_fakes.trust_id, identity_fakes.trust_expires, identity_fakes.trust_impersonation, identity_fakes.project_id, identity_fakes.user_id, identity_fakes.user_id),)
        self.assertEqual(datalist, tuple(data))

    def test_trust_list_trustee(self):
        arglist = ['--trustee', identity_fakes.user_name]
        verifylist = [('trustor', None), ('trustee', identity_fakes.user_name), ('authuser', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.trusts_mock.list.assert_any_call(trustee_user=self.users_mock.get(), trustor_user=None)
        collist = ('ID', 'Expires At', 'Impersonation', 'Project ID', 'Trustee User ID', 'Trustor User ID')
        self.assertEqual(collist, columns)
        datalist = ((identity_fakes.trust_id, identity_fakes.trust_expires, identity_fakes.trust_impersonation, identity_fakes.project_id, identity_fakes.user_id, identity_fakes.user_id),)
        self.assertEqual(datalist, tuple(data))

    def test_trust_list_trustor(self):
        arglist = ['--trustor', identity_fakes.user_name]
        verifylist = [('trustee', None), ('trustor', identity_fakes.user_name), ('authuser', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.trusts_mock.list.assert_any_call(trustor_user=self.users_mock.get(), trustee_user=None)
        collist = ('ID', 'Expires At', 'Impersonation', 'Project ID', 'Trustee User ID', 'Trustor User ID')
        self.assertEqual(collist, columns)
        datalist = ((identity_fakes.trust_id, identity_fakes.trust_expires, identity_fakes.trust_impersonation, identity_fakes.project_id, identity_fakes.user_id, identity_fakes.user_id),)
        self.assertEqual(datalist, tuple(data))