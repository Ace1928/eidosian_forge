import json
from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_replica_export_location_list(self):
    slug = 'replica-supported'
    share_type = self.create_share_type(data_utils.rand_name(slug), 'False', extra_specs={'replication_type': 'readable'})
    share = self.create_share(share_type=share_type['name'])
    replica = self.create_share_replica(share['id'], wait=True)
    rep_exp_loc_list = self.listing_result('share replica export location', f'list {replica['id']}')
    self.assertTableStruct(rep_exp_loc_list, ['ID', 'Availability Zone', 'Replica State', 'Preferred', 'Path'])
    exp_loc_list = self.openstack(f'share replica show {replica['id']} -f json')
    exp_loc_list = json.loads(exp_loc_list)
    self.assertIn(exp_loc_list.get('export_locations')[0]['id'], [item['ID'] for item in rep_exp_loc_list])