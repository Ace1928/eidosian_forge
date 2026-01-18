import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
def test_volume_target_show_fields(self):
    volume_target = self.mgr.get(TARGET1['uuid'], fields=['uuid', 'boot_index'])
    expect = [('GET', '/v1/volume/targets/%s?fields=uuid,boot_index' % TARGET1['uuid'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(TARGET1['uuid'], volume_target.uuid)
    self.assertEqual(TARGET1['boot_index'], volume_target.boot_index)