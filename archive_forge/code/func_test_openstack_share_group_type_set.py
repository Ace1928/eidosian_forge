from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from manilaclient.tests.functional.osc import base
def test_openstack_share_group_type_set(self):
    share_group_type_name = data_utils.rand_name('test_share_group_type_create')
    share_type_name = 'dhss_false'
    share_group_type = self.create_share_group_type(name=share_group_type_name, share_types=share_type_name)
    shares_group_type_show = self.openstack(f'share group type show {share_group_type_name} -f json')
    shares_group_type_show = jsonutils.loads(shares_group_type_show)
    expected_sgt_values = {'id': share_group_type['id'], 'group_specs': {}}
    for k, v in expected_sgt_values.items():
        self.assertEqual(expected_sgt_values[k], shares_group_type_show[k])
    group_snap_key = 'snapshot_support'
    group_snap_value = 'False'
    group_specs = f'{group_snap_key}={group_snap_value}'
    self.dict_result('share', f'group type set {share_group_type_name} --group-specs {group_specs}')
    shares_group_type_show = self.openstack(f'share group type show {share_group_type_name} -f json')
    shares_group_type_show = jsonutils.loads(shares_group_type_show)
    expected_sgt_values = {'id': share_group_type['id'], 'group_specs': {group_snap_key: group_snap_value}}
    for k, v in expected_sgt_values.items():
        self.assertEqual(expected_sgt_values[k], shares_group_type_show[k])