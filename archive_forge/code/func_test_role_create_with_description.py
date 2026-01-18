import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_create_with_description(self):
    self.roles_mock.create.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
    arglist = ['--description', identity_fakes.role_description, identity_fakes.ROLE_2['name']]
    verifylist = [('description', identity_fakes.role_description), ('name', identity_fakes.ROLE_2['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'description': identity_fakes.role_description, 'name': identity_fakes.ROLE_2['name'], 'domain': None, 'options': {}}
    self.roles_mock.create.assert_called_with(**kwargs)
    collist = ('domain', 'id', 'name')
    self.assertEqual(collist, columns)
    datalist = ('d1', identity_fakes.ROLE_2['id'], identity_fakes.ROLE_2['name'])
    self.assertEqual(datalist, data)