import copy
from openstackclient.identity.v3 import federation_protocol
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestProtocolShow(TestProtocol):

    def setUp(self):
        super(TestProtocolShow, self).setUp()
        self.protocols_mock.get.return_value = fakes.FakeResource(None, identity_fakes.PROTOCOL_OUTPUT, loaded=False)
        self.cmd = federation_protocol.ShowProtocol(self.app, None)

    def test_show_protocol(self):
        arglist = [identity_fakes.protocol_id, '--identity-provider', identity_fakes.idp_id]
        verifylist = [('federation_protocol', identity_fakes.protocol_id), ('identity_provider', identity_fakes.idp_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.protocols_mock.get.assert_called_with(identity_fakes.idp_id, identity_fakes.protocol_id)
        collist = ('id', 'identity_provider', 'mapping')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.protocol_id, identity_fakes.idp_id, identity_fakes.mapping_id)
        self.assertEqual(datalist, data)