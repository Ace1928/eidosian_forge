import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_targets_list_pagination_no_limit(self):
    volume_targets = self.mgr.list(limit=0)
    expect = [('GET', '/v1/volume/targets', {}, None), ('GET', '/v1/volume/targets/?marker=%s' % TARGET1['uuid'], {}, None)]
    expect_targets = [TARGET1, TARGET2]
    self._validate_list(expect, expect_targets, volume_targets)