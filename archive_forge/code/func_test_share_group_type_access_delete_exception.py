from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import share_group_type_access as osc_sgta
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_type_access_delete_exception(self):
    arglist = [self.share_group_type.id, 'invalid_project_format']
    verifylist = [('share_group_type', self.share_group_type.id), ('projects', ['invalid_project_format'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.type_access_mock.remove_project_access.side_effect = BadRequest()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)