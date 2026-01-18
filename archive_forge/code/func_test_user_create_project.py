from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_create_project(self):
    self.projects_mock.get.return_value = self.fake_project_c
    attr = {'tenantId': self.fake_project_c.id}
    user_2 = identity_fakes.FakeUser.create_one_user(attr)
    self.users_mock.create.return_value = user_2
    arglist = ['--project', self.fake_project_c.name, user_2.name]
    verifylist = [('name', user_2.name), ('project', self.fake_project_c.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': True, 'tenant_id': self.fake_project_c.id}
    self.users_mock.create.assert_called_with(user_2.name, None, None, **kwargs)
    self.assertEqual(self.columns, columns)
    datalist = (user_2.email, True, user_2.id, user_2.name, self.fake_project_c.id)
    self.assertEqual(datalist, data)