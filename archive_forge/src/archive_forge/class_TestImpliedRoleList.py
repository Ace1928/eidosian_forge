import copy
from openstackclient.identity.v3 import implied_role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestImpliedRoleList(TestRole):

    def setUp(self):
        super(TestImpliedRoleList, self).setUp()
        self.inference_rules_mock.list_inference_roles.return_value = identity_fakes.FakeImpliedRoleResponse.create_list()
        self.cmd = implied_role.ListImpliedRole(self.app, None)

    def test_implied_role_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.inference_rules_mock.list_inference_roles.assert_called_with()
        collist = ['Prior Role ID', 'Prior Role Name', 'Implied Role ID', 'Implied Role Name']
        self.assertEqual(collist, columns)
        datalist = [(identity_fakes.ROLES[0]['id'], identity_fakes.ROLES[0]['name'], identity_fakes.ROLES[1]['id'], identity_fakes.ROLES[1]['name'])]
        x = [d for d in data]
        self.assertEqual(datalist, x)