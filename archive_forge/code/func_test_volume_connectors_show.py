import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_connectors_show(self):
    volume_connector = self.mgr.get(CONNECTOR1['uuid'])
    expect = [('GET', '/v1/volume/connectors/%s' % CONNECTOR1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self._validate_obj(CONNECTOR1, volume_connector)