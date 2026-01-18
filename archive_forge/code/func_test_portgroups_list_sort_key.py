import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_list_sort_key(self):
    portgroups = self.mgr.list(sort_key='updated_at')
    expect = [('GET', '/v1/portgroups/?sort_key=updated_at', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(portgroups))
    expected_resp = ({}, {'portgroups': [PORTGROUP2, PORTGROUP]})
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/?sort_key=updated_at']['GET'])