import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_ports_list_by_node(self):
    ports = self.mgr.list(node=PORT['node_uuid'])
    expect = [('GET', '/v1/ports/?node=%s' % PORT['node_uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(ports))