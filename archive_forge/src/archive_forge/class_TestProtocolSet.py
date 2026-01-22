import copy
from openstackclient.identity.v3 import federation_protocol
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
class TestProtocolSet(TestProtocol):

    def setUp(self):
        super(TestProtocolSet, self).setUp()
        self.protocols_mock.get.return_value = fakes.FakeResource(None, identity_fakes.PROTOCOL_OUTPUT, loaded=True)
        self.protocols_mock.update.return_value = fakes.FakeResource(None, identity_fakes.PROTOCOL_OUTPUT_UPDATED, loaded=True)
        self.cmd = federation_protocol.SetProtocol(self.app, None)

    def test_set_new_mapping(self):
        arglist = [identity_fakes.protocol_id, '--identity-provider', identity_fakes.idp_id, '--mapping', identity_fakes.mapping_id]
        verifylist = [('identity_provider', identity_fakes.idp_id), ('federation_protocol', identity_fakes.protocol_id), ('mapping', identity_fakes.mapping_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.protocols_mock.update.assert_called_with(identity_fakes.idp_id, identity_fakes.protocol_id, identity_fakes.mapping_id)
        collist = ('id', 'identity_provider', 'mapping')
        self.assertEqual(collist, columns)
        datalist = (identity_fakes.protocol_id, identity_fakes.idp_id, identity_fakes.mapping_id_updated)
        self.assertEqual(datalist, data)