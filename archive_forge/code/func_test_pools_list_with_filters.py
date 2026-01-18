from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.tests.functional import base
def test_pools_list_with_filters(self):
    self.clients['admin'].manila('pool-list', params='--host myhost --backend mybackend --pool mypool')