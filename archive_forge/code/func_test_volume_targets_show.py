import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_targets_show(self):
    volume_target = self.mgr.get(TARGET1['uuid'])
    expect = [('GET', '/v1/volume/targets/%s' % TARGET1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self._validate_obj(TARGET1, volume_target)