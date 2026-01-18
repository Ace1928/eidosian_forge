from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_show(self):
    arglist = [self.fake_user.id]
    verifylist = [('user', self.fake_user.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.users_mock.get.assert_called_with(self.fake_user.id)
    collist = ('email', 'enabled', 'id', 'name', 'project_id')
    self.assertEqual(collist, columns)
    datalist = (self.fake_user.email, True, self.fake_user.id, self.fake_user.name, self.fake_project.id)
    self.assertCountEqual(datalist, data)