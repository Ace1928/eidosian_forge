from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import ndp_proxy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestDeleteNDPProxy(TestNDPProxy):

    def setUp(self):
        super(TestDeleteNDPProxy, self).setUp()
        attrs = {'router_id': self.router.id, 'port_id': self.port.id}
        self.ndp_proxies = network_fakes.create_ndp_proxies(attrs)
        self.ndp_proxy = self.ndp_proxies[0]
        self.network_client.delete_ndp_proxy = mock.Mock(return_value=None)
        self.network_client.find_ndp_proxy = mock.Mock(return_value=self.ndp_proxy)
        self.cmd = ndp_proxy.DeleteNDPProxy(self.app, self.namespace)

    def test_delete(self):
        arglist = [self.ndp_proxy.id]
        verifylist = [('ndp_proxy', [self.ndp_proxy.id])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_ndp_proxy.assert_called_once_with(self.ndp_proxy)
        self.assertIsNone(result)

    def test_delete_error(self):
        arglist = [self.ndp_proxy.id]
        verifylist = [('ndp_proxy', [self.ndp_proxy.id])]
        self.network_client.delete_ndp_proxy.side_effect = Exception('Error message')
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_multi_ndp_proxies_delete(self):
        arglist = []
        np_id = []
        for a in self.ndp_proxies:
            arglist.append(a.id)
            np_id.append(a.id)
        verifylist = [('ndp_proxy', np_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_ndp_proxy.assert_has_calls([call(self.ndp_proxy), call(self.ndp_proxy)])
        self.assertIsNone(result)