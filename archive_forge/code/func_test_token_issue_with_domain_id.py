from unittest import mock
from openstackclient.identity.v3 import token
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_token_issue_with_domain_id(self):
    auth_ref = identity_fakes.fake_auth_ref(identity_fakes.TOKEN_WITH_DOMAIN_ID)
    self.ar_mock = mock.PropertyMock(return_value=auth_ref)
    type(self.app.client_manager).auth_ref = self.ar_mock
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    collist = ('domain_id', 'expires', 'id', 'user_id')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.domain_id, identity_fakes.token_expires, identity_fakes.token_id, identity_fakes.user_id)
    self.assertEqual(datalist, data)