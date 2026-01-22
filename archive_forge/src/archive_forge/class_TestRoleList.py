from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import role
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
class TestRoleList(TestRole):

    def setUp(self):
        super(TestRoleList, self).setUp()
        self.roles_mock.list.return_value = [self.fake_role]
        self.cmd = role.ListRole(self.app, None)

    def test_role_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.roles_mock.list.assert_called_with()
        collist = ('ID', 'Name')
        self.assertEqual(collist, columns)
        datalist = ((self.fake_role.id, self.fake_role.name),)
        self.assertEqual(datalist, tuple(data))