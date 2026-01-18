import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def wait_for_share_server_deletion(self, share_server, microversion=None):
    """Wait for share server deletion by its Name or ID.

        :param share_server: text -- Name or ID of share server
        """
    self.wait_for_resource_deletion(SHARE_SERVER, res_id=share_server, interval=3, timeout=60, microversion=microversion)