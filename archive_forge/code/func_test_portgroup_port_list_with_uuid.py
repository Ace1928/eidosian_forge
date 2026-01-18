import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroup_port_list_with_uuid(self):
    ports = self.mgr.list_ports(PORTGROUP['uuid'])
    expect = [('GET', '/v1/portgroups/%s/ports' % PORTGROUP['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(ports))
    self.assertEqual(PORT['uuid'], ports[0].uuid)
    self.assertEqual(PORT['address'], ports[0].address)
    expected_resp = ({}, {'ports': [PORT]})
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/%s/ports' % PORTGROUP['uuid']]['GET'])