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