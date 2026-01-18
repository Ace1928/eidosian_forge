from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import share_type_access as osc_share_type_access
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_type_access_create(self):
    arglist = [self.share_type.id, self.project.id]
    verifylist = [('share_type', self.share_type.id), ('project_id', self.project.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.type_access_mock.add_project_access.assert_called_with(self.share_type, self.project.id)
    self.assertIsNone(result)