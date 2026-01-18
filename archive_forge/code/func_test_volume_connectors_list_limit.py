import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_connectors_list_limit(self):
    volume_connectors = self.mgr.list(limit=1)
    expect = [('GET', '/v1/volume/connectors/?limit=1', {}, None)]
    expect_connectors = [CONNECTOR1]
    self._validate_list(expect, expect_connectors, volume_connectors)