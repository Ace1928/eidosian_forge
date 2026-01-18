import copy
import testtools
from testtools.matchers import HasLength
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_ports_list_marker(self):
    self.api = utils.FakeAPI(fake_responses_pagination)
    self.mgr = ironicclient.v1.port.PortManager(self.api)
    ports = self.mgr.list(marker=PORT['uuid'])
    expect = [('GET', '/v1/ports/?marker=%s' % PORT['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(ports, HasLength(1))