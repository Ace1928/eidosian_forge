import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_ports_list_by_address_detail(self):
    ports = self.mgr.list(address=PORT['address'], detail=True)
    expect = [('GET', '/v1/ports/detail?address=%s' % PORT['address'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(ports))