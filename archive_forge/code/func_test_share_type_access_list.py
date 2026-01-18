from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.osc.v2 import share_type_access as osc_share_type_access
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_type_access_list(self):
    share_type = manila_fakes.FakeShareType.create_one_sharetype(attrs={'share_type_access:is_public': False})
    self.share_types_mock.get.return_value = share_type
    arglist = [share_type.id]
    verifylist = [('share_type', share_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.type_access_mock.list.assert_called_once_with(share_type)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, tuple(data))