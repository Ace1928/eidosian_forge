import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_ports_list_sort_key(self):
    self.api = utils.FakeAPI(fake_responses_sorting)
    self.mgr = ironicclient.v1.port.PortManager(self.api)
    ports = self.mgr.list(sort_key='updated_at')
    expect = [('GET', '/v1/ports/?sort_key=updated_at', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(2, len(ports))