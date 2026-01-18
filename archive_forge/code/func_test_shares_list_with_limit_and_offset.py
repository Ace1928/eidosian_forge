import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data('admin', 'user')
def test_shares_list_with_limit_and_offset(self, role):
    self.clients[role].manila('list', params='--limit 1 --offset 1')