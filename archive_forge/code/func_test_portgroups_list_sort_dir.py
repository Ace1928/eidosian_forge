import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_list_sort_dir(self):
    portgroups = self.mgr.list(sort_dir='desc')
    expect = [('GET', '/v1/portgroups/?sort_dir=desc', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(portgroups))
    expected_resp = ({}, {'portgroups': [PORTGROUP2, PORTGROUP]})
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups/?sort_dir=desc']['GET'])