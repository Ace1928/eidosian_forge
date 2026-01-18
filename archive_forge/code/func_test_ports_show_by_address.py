import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_ports_show_by_address(self):
    port = self.mgr.get_by_address(PORT['address'])
    expect = [('GET', '/v1/ports/detail?address=%s' % PORT['address'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(PORT['uuid'], port.uuid)
    self.assertEqual(PORT['address'], port.address)
    self.assertEqual(PORT['node_uuid'], port.node_uuid)
    self.assertEqual(PORT['pxe_enabled'], port.pxe_enabled)
    self.assertEqual(PORT['local_link_connection'], port.local_link_connection)
    self.assertEqual(PORT['portgroup_uuid'], port.portgroup_uuid)
    self.assertEqual(PORT['physical_network'], port.physical_network)
    self.assertEqual(PORT['is_smartnic'], port.is_smartnic)