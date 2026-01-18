import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_list_marker(self):
    portgroups = self.mgr.list(marker=PORTGROUP['uuid'])
    expect = [('GET', '/v1/portgroups/?marker=%s' % PORTGROUP['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(portgroups))
    expected_resp = ({}, {'next': 'http://127.0.0.1:6385/v1/portgroups/?limit=1', 'portgroups': [PORTGROUP]})
    self.assertEqual(expected_resp, self.api.responses['/v1/portgroups']['GET'])