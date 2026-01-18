import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_ports_list_pagination_no_limit(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = ironicclient.v1.port.PortManager(self.api)
    ports = self.mgr.list(limit=0)
    expect = [('GET', '/v1/ports', {}, None), ('GET', '/v1/ports/?limit=1', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(ports, HasLength(2))