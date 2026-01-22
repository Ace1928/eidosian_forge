import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
def _verify_access(self, share_type_id, is_public, microversion=None):
    if is_public:
        share_types = self.admin_client.list_share_types(list_all=False, microversion=microversion)
        self.assertTrue(any((share_type_id == st['ID'] for st in share_types)))
    else:
        self.assertFalse(self._share_type_listed_by(share_type_id=share_type_id, by_admin=False, list_all=True, microversion=microversion))
        self.assertTrue(self._share_type_listed_by(share_type_id=share_type_id, by_admin=True, list_all=True, microversion=microversion))
        self.assertFalse(self._share_type_listed_by(share_type_id=share_type_id, by_admin=True, list_all=False, microversion=microversion))