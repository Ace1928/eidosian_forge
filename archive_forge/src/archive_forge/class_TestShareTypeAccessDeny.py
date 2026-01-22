from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import share_type_access as osc_share_type_access
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareTypeAccessDeny(TestShareTypeAccess):

    def setUp(self):
        super(TestShareTypeAccessDeny, self).setUp()
        self.project = identity_fakes.FakeProject.create_one_project()
        self.share_type = manila_fakes.FakeShareType.create_one_sharetype(attrs={'share_type_access:is_public': False})
        self.share_types_mock.get.return_value = self.share_type
        self.type_access_mock.remove_project_access.return_value = None
        self.cmd = osc_share_type_access.ShareTypeAccessDeny(self.app, None)

    def test_share_type_access_delete(self):
        arglist = [self.share_type.id, self.project.id]
        verifylist = [('share_type', self.share_type.id), ('project_id', self.project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.type_access_mock.remove_project_access.assert_called_with(self.share_type, self.project.id)
        self.assertIsNone(result)

    def test_share_type_access_delete_exception(self):
        arglist = [self.share_type.id, 'invalid_project_format']
        verifylist = [('share_type', self.share_type.id), ('project_id', 'invalid_project_format')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.type_access_mock.remove_project_access.side_effect = BadRequest()
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)