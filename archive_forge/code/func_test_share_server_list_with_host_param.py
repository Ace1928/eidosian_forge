import ast
import ddt
import testtools
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_share_server_list_with_host_param(self):
    self.client.list_share_servers(filters={'host': 'fake_host'})