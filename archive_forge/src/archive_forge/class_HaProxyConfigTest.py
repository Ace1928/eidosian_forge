import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.aws.lb import loadbalancer as lb
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class HaProxyConfigTest(common.HeatTestCase):

    def setUp(self):
        super(HaProxyConfigTest, self).setUp()
        self.stack = utils.parse_stack(template_format.parse(lb_template))
        resource_name = 'LoadBalancer'
        lb_defn = self.stack.t.resource_definitions(self.stack)[resource_name]
        self.lb = lb.LoadBalancer(resource_name, lb_defn, self.stack)
        self.lb.client_plugin = mock.Mock()

    def _mock_props(self, props):

        def get_props(name):
            return props[name]
        self.lb.properties = mock.MagicMock()
        self.lb.properties.__getitem__.side_effect = get_props

    def test_combined(self):
        self.lb._haproxy_config_global = mock.Mock(return_value='one,')
        self.lb._haproxy_config_frontend = mock.Mock(return_value='two,')
        self.lb._haproxy_config_backend = mock.Mock(return_value='three,')
        self.lb._haproxy_config_servers = mock.Mock(return_value='four')
        actual = self.lb._haproxy_config([3, 5])
        self.assertEqual('one,two,three,four\n', actual)
        self.lb._haproxy_config_global.assert_called_once_with()
        self.lb._haproxy_config_frontend.assert_called_once_with()
        self.lb._haproxy_config_backend.assert_called_once_with()
        self.lb._haproxy_config_servers.assert_called_once_with([3, 5])

    def test_global(self):
        exp = '\nglobal\n    daemon\n    maxconn 256\n    stats socket /tmp/.haproxy-stats\n\ndefaults\n    mode http\n    timeout connect 5000ms\n    timeout client 50000ms\n    timeout server 50000ms\n'
        actual = self.lb._haproxy_config_global()
        self.assertEqual(exp, actual)

    def test_frontend(self):
        props = {'HealthCheck': {}, 'Listeners': [{'LoadBalancerPort': 4014}]}
        self._mock_props(props)
        exp = '\nfrontend http\n    bind *:4014\n    default_backend servers\n'
        actual = self.lb._haproxy_config_frontend()
        self.assertEqual(exp, actual)

    def test_backend_with_timeout(self):
        props = {'HealthCheck': {'Timeout': 43}}
        self._mock_props(props)
        actual = self.lb._haproxy_config_backend()
        exp = '\nbackend servers\n    balance roundrobin\n    option http-server-close\n    option forwardfor\n    option httpchk\n    timeout check 43s\n'
        self.assertEqual(exp, actual)

    def test_backend_no_timeout(self):
        self._mock_props({'HealthCheck': None})
        be = self.lb._haproxy_config_backend()
        exp = '\nbackend servers\n    balance roundrobin\n    option http-server-close\n    option forwardfor\n    option httpchk\n\n'
        self.assertEqual(exp, be)

    def test_servers_none(self):
        props = {'HealthCheck': {}, 'Listeners': [{'InstancePort': 1234}]}
        self._mock_props(props)
        actual = self.lb._haproxy_config_servers([])
        exp = ''
        self.assertEqual(exp, actual)

    def test_servers_no_check(self):
        props = {'HealthCheck': {}, 'Listeners': [{'InstancePort': 4511}]}
        self._mock_props(props)

        def fake_to_ipaddr(inst):
            return '192.168.1.%s' % inst
        to_ip = self.lb.client_plugin.return_value.server_to_ipaddress
        to_ip.side_effect = fake_to_ipaddr
        actual = self.lb._haproxy_config_servers(range(1, 3))
        exp = '\n    server server1 192.168.1.1:4511\n    server server2 192.168.1.2:4511'
        self.assertEqual(exp.replace('\n', '', 1), actual)

    def test_servers_servers_and_check(self):
        props = {'HealthCheck': {'HealthyThreshold': 1, 'Interval': 2, 'Target': 'HTTP:80/', 'Timeout': 45, 'UnhealthyThreshold': 5}, 'Listeners': [{'InstancePort': 1234}]}
        self._mock_props(props)

        def fake_to_ipaddr(inst):
            return '192.168.1.%s' % inst
        to_ip = self.lb.client_plugin.return_value.server_to_ipaddress
        to_ip.side_effect = fake_to_ipaddr
        actual = self.lb._haproxy_config_servers(range(1, 3))
        exp = '\n    server server1 192.168.1.1:1234 check inter 2s fall 5 rise 1\n    server server2 192.168.1.2:1234 check inter 2s fall 5 rise 1'
        self.assertEqual(exp.replace('\n', '', 1), actual)