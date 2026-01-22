from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_pair
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestDeleteSfcPortPair(fakes.TestNeutronClientOSCV2):
    _port_pair = fakes.FakeSfcPortPair.create_port_pairs(count=1)

    def setUp(self):
        super(TestDeleteSfcPortPair, self).setUp()
        self.network.delete_sfc_port_pair = mock.Mock(return_value=None)
        self.cmd = sfc_port_pair.DeleteSfcPortPair(self.app, self.namespace)

    def test_delete_port_pair(self):
        client = self.app.client_manager.network
        mock_port_pair_delete = client.delete_sfc_port_pair
        arglist = [self._port_pair[0]['id']]
        verifylist = [('port_pair', [self._port_pair[0]['id']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        mock_port_pair_delete.assert_called_once_with(self._port_pair[0]['id'])
        self.assertIsNone(result)

    def test_delete_multiple_port_pairs_with_exception(self):
        client = self.app.client_manager.network
        target1 = self._port_pair[0]['id']
        arglist = [target1]
        verifylist = [('port_pair', [target1])]
        client.find_sfc_port_pair.side_effect = [target1, exceptions.CommandError]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        msg = '1 of 2 port pair(s) failed to delete.'
        with testtools.ExpectedException(exceptions.CommandError) as e:
            self.cmd.take_action(parsed_args)
            self.assertEqual(msg, str(e))