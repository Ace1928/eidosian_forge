import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_port_list_fields(self):
    ports = self.mgr.list(fields=['uuid', 'address'])
    expect = [('GET', '/v1/ports/?fields=uuid,address', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(ports))