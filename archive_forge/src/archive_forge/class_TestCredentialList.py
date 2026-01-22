from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestCredentialList(TestCredential):
    credential = identity_fakes.FakeCredential.create_one_credential()
    columns = ('ID', 'Type', 'User ID', 'Data', 'Project ID')
    data = ((credential.id, credential.type, credential.user_id, credential.blob, credential.project_id),)

    def setUp(self):
        super(TestCredentialList, self).setUp()
        self.user = identity_fakes.FakeUser.create_one_user()
        self.users_mock.get.return_value = self.user
        self.credentials_mock.list.return_value = [self.credential]
        self.cmd = credential.ListCredential(self.app, None)

    def test_credential_list_no_options(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.credentials_mock.list.assert_called_with()
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))

    def test_credential_list_with_options(self):
        arglist = ['--user', self.credential.user_id, '--type', self.credential.type]
        verifylist = [('user', self.credential.user_id), ('type', self.credential.type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'user_id': self.user.id, 'type': self.credential.type}
        self.users_mock.get.assert_called_with(self.credential.user_id)
        self.credentials_mock.list.assert_called_with(**kwargs)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, tuple(data))