import copy
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_role_set_domain_role(self):
    self.roles_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.ROLE_2), loaded=True)
    arglist = ['--name', 'over', '--domain', identity_fakes.domain_name, identity_fakes.ROLE_2['name']]
    verifylist = [('name', 'over'), ('domain', identity_fakes.domain_name), ('role', identity_fakes.ROLE_2['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'name': 'over', 'description': None, 'options': {}}
    self.roles_mock.update.assert_called_with(identity_fakes.ROLE_2['id'], **kwargs)
    self.assertIsNone(result)