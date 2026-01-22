from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.identity.v3 import credential
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils
class TestCredentialShow(TestCredential):
    columns = ('blob', 'id', 'project_id', 'type', 'user_id')

    def setUp(self):
        super(TestCredentialShow, self).setUp()
        self.credential = identity_fakes.FakeCredential.create_one_credential()
        self.credentials_mock.get.return_value = self.credential
        self.data = (self.credential.blob, self.credential.id, self.credential.project_id, self.credential.type, self.credential.user_id)
        self.cmd = credential.ShowCredential(self.app, None)

    def test_credential_show(self):
        arglist = [self.credential.id]
        verifylist = [('credential', self.credential.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.credentials_mock.get.assert_called_once_with(self.credential.id)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)