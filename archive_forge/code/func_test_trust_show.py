import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v3 import trust
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_trust_show(self):
    arglist = [identity_fakes.trust_id]
    verifylist = [('trust', identity_fakes.trust_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.trusts_mock.get.assert_called_with(identity_fakes.trust_id)
    collist = ('expires_at', 'id', 'impersonation', 'project_id', 'roles', 'trustee_user_id', 'trustor_user_id')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.trust_expires, identity_fakes.trust_id, identity_fakes.trust_impersonation, identity_fakes.project_id, identity_fakes.role_name, identity_fakes.user_id, identity_fakes.user_id)
    self.assertEqual(datalist, data)