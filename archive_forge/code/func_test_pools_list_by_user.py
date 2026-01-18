from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.tests.functional import base
def test_pools_list_by_user(self):
    self.assertRaises(exceptions.CommandFailed, self.clients['user'].manila, 'pool-list')