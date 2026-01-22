import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
def _share_type_listed_by(self, share_type_id, by_admin=False, list_all=False, microversion=None):
    client = self.admin_client if by_admin else self.user_client
    share_types = client.list_share_types(list_all=list_all, microversion=microversion)
    return any((share_type_id == st['ID'] for st in share_types))