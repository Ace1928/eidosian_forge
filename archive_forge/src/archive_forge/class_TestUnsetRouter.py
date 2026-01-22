from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestUnsetRouter(TestRouter):

    def setUp(self):
        super(TestUnsetRouter, self).setUp()
        self.fake_network = network_fakes.create_one_network()
        self.fake_qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self._testrouter = network_fakes.FakeRouter.create_one_router({'routes': [{'destination': '192.168.101.1/24', 'nexthop': '172.24.4.3'}, {'destination': '192.168.101.2/24', 'nexthop': '172.24.4.3'}], 'tags': ['green', 'red'], 'external_gateway_info': {'network_id': self.fake_network.id, 'qos_policy_id': self.fake_qos_policy.id}})
        self.fake_subnet = network_fakes.FakeSubnet.create_one_subnet()
        self.network_client.find_router = mock.Mock(return_value=self._testrouter)
        self.network_client.update_router = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.cmd = router.UnsetRouter(self.app, self.namespace)

    def test_unset_router_params(self):
        arglist = ['--route', 'destination=192.168.101.1/24,gateway=172.24.4.3', self._testrouter.name]
        verifylist = [('routes', [{'destination': '192.168.101.1/24', 'gateway': '172.24.4.3'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'routes': [{'destination': '192.168.101.2/24', 'nexthop': '172.24.4.3'}]}
        self.network_client.update_router.assert_called_once_with(self._testrouter, **attrs)
        self.assertIsNone(result)

    def test_unset_router_wrong_routes(self):
        arglist = ['--route', 'destination=192.168.101.1/24,gateway=172.24.4.2', self._testrouter.name]
        verifylist = [('routes', [{'destination': '192.168.101.1/24', 'gateway': '172.24.4.2'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_router_external_gateway(self):
        arglist = ['--external-gateway', self._testrouter.name]
        verifylist = [('external_gateway', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'external_gateway_info': {}}
        self.network_client.update_router.assert_called_once_with(self._testrouter, **attrs)
        self.assertIsNone(result)

    def _test_unset_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['green']
        else:
            arglist = ['--all-tag']
            verifylist = [('all_tag', True)]
            expected_args = []
        arglist.append(self._testrouter.name)
        verifylist.append(('router', self._testrouter.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_router.called)
        self.network_client.set_tags.assert_called_once_with(self._testrouter, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_unset_with_tags(self):
        self._test_unset_tags(with_tags=True)

    def test_unset_with_all_tag(self):
        self._test_unset_tags(with_tags=False)

    def test_unset_router_qos_policy(self):
        arglist = ['--qos-policy', self._testrouter.name]
        verifylist = [('qos_policy', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'external_gateway_info': {'network_id': self.fake_network.id, 'qos_policy_id': None}}
        self.network_client.update_router.assert_called_once_with(self._testrouter, **attrs)
        self.assertIsNone(result)

    def test_unset_gateway_ip_qos_no_network(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        router = network_fakes.FakeRouter.create_one_router()
        self.network_client.find_router = mock.Mock(return_value=router)
        arglist = ['--qos-policy', router.id]
        verifylist = [('router', router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_gateway_ip_qos_no_qos(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        router = network_fakes.FakeRouter.create_one_router({'external_gateway_info': {'network_id': 'fake-id'}})
        self.network_client.find_router = mock.Mock(return_value=router)
        arglist = ['--qos-policy', router.id]
        verifylist = [('router', router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)