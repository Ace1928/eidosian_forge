import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_show(self):
    portgroup = self.mgr.get(PORTGROUP['uuid'])
    expect = [('GET', '/v1/portgroups/%s' % PORTGROUP['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(PORTGROUP['uuid'], portgroup.uuid)
    self.assertEqual(PORTGROUP['name'], portgroup.name)
    self.assertEqual(PORTGROUP['node_uuid'], portgroup.node_uuid)
    self.assertEqual(PORTGROUP['address'], portgroup.address)
    expected_resp = ({}, PORTGROUP)
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/%s' % PORTGROUP['uuid']]['GET'])