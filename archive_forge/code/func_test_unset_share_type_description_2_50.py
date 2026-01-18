import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
def test_unset_share_type_description_2_50(self):
    self.skip_if_microversion_not_supported('2.50')
    microversion = '2.50'
    share_type_name = data_utils.rand_name('share_type_update_test')
    share_type = self.create_share_type(name=share_type_name, driver_handles_share_servers=False, snapshot_support=None, create_share_from_snapshot=None, revert_to_snapshot=None, mount_snapshot=None, is_public=True, microversion=microversion, extra_specs={}, description='share_type_description')
    st_id = share_type['ID']
    new_description = ''
    st_updated = self.update_share_type(st_id, description=new_description, microversion=microversion)
    self.assertEqual('None', st_updated['Description'])
    self.admin_client.delete_share_type(st_id, microversion=microversion)
    self.admin_client.wait_for_share_type_deletion(st_id, microversion=microversion)
    share_types = self.admin_client.list_share_types(list_all=False, microversion=microversion)
    self.assertFalse(any((st_id == st['ID'] for st in share_types)))