import json
from manilaclient.tests.functional.osc import base
def test_share_type_unset(self):
    share_type = self.create_share_type(snapshot_support=True, extra_specs={'foo': 'bar'})
    self.openstack(f'share type unset {share_type['id']} snapshot_support foo')
    share_type = json.loads(self.openstack(f'share type show {share_type['id']} -f json'))
    self.assertNotIn('foo', share_type['optional_extra_specs'])
    self.assertNotIn('snapshot_support', share_type['optional_extra_specs'])