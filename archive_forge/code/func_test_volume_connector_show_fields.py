import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_connector_show_fields(self):
    volume_connector = self.mgr.get(CONNECTOR1['uuid'], fields=['uuid', 'connector_id'])
    expect = [('GET', '/v1/volume/connectors/%s?fields=uuid,connector_id' % CONNECTOR1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(CONNECTOR1['uuid'], volume_connector.uuid)
    self.assertEqual(CONNECTOR1['connector_id'], volume_connector.connector_id)