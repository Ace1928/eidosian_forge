from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import role
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_role_add(self):
    arglist = ['--project', self.fake_project.name, '--user', self.fake_user.name, self.fake_role.name]
    verifylist = [('project', self.fake_project.name), ('user', self.fake_user.name), ('role', self.fake_role.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.roles_mock.add_user_role.assert_called_with(self.fake_user.id, self.fake_role.id, self.fake_project.id)
    collist = ('id', 'name')
    self.assertEqual(collist, columns)
    datalist = (self.fake_role.id, self.fake_role.name)
    self.assertEqual(datalist, data)