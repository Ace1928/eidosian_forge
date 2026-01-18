import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_list(self):
    portgroups = self.mgr.list()
    expect = [('GET', '/v1/portgroups', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(portgroups))
    expected_resp = ({}, {'portgroups': [PORTGROUP, PORTGROUP2]})
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups']['GET'])