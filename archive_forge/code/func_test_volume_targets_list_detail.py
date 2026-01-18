import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_targets_list_detail(self):
    volume_targets = self.mgr.list(detail=True)
    expect = [('GET', '/v1/volume/targets/?detail=True', {}, None)]
    expect_targets = [TARGET1]
    self._validate_list(expect, expect_targets, volume_targets)