import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_connector_list_fields(self):
    volume_connectors = self.mgr.list(fields=['uuid', 'connector_id'])
    expect = [('GET', '/v1/volume/connectors/?fields=uuid,connector_id', {}, None)]
    expect_connectors = [CONNECTOR1]
    self._validate_list(expect, expect_connectors, volume_connectors)