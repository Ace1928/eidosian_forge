import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data('admin', 'user')
def test_shares_list_filter_by_export_location(self, role):
    self.clients[role].manila('list', params='--export_location fake')