from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_revert(self):
    slug = 'revert_test'
    share_type = self.create_share_type(name=data_utils.rand_name(slug), snapshot_support=True, revert_to_snapshot_support=True)
    share = self.create_share(share_type=share_type['id'], size=10)
    snapshot = self.create_snapshot(share['id'], wait=True)
    self.assertEqual(snapshot['size'], share['size'])
    self.openstack(f'share resize {share['id']} 15 --wait')
    result1 = self.dict_result('share', f'show {share['id']}')
    self.assertEqual('15', result1['size'])
    self.openstack(f'share revert {snapshot['id']} --wait')
    result2 = self.dict_result('share', f'show {share['id']}')
    self.assertEqual('10', result2['size'])