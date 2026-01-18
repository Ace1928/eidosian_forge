import copy
import testtools
from ironicclient.tests.unit import utils
import ironicclient.v1.portgroup
def test_portgroups_list_by_address_detail(self):
    portgroups = self.mgr.list(address=PORTGROUP['address'], detail=True)
    expect = [('GET', '/v1/portgroups/detail?address=%s' % PORTGROUP['address'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(portgroups))
    self.assertIn(PORTGROUP, self.api.responses['/v1/portgroups']['GET'][1]['portgroups'])