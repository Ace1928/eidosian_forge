import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_show_domain_role(self):
    self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
    arglist = ['--domain', identity_fakes.domain_name, identity_fakes.ROLE_2['name']]
    verifylist = [('domain', identity_fakes.domain_name), ('role', identity_fakes.ROLE_2['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.roles_mock.get.assert_called_with(identity_fakes.ROLE_2['name'])
    collist = ('domain', 'id', 'name')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.domain_id, identity_fakes.ROLE_2['id'], identity_fakes.ROLE_2['name'])
    self.assertEqual(datalist, data)