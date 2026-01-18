import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
import testtools
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
def test_shares_list_filter_by_share_server_as_admin(self):
    self.clients['admin'].manila('list', params='--share-server fake')