import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_targets_list_sort_dir(self):
    volume_targets = self.mgr.list(sort_dir='desc')
    expect = [('GET', '/v1/volume/targets/?sort_dir=desc', {}, None)]
    expect_targets = [TARGET2, TARGET1]
    self._validate_list(expect, expect_targets, volume_targets)