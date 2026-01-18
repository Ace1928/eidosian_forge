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
def wait_for_share_replica_deletion(self, replica, microversion=None):
    """Wait for share replica deletion by its ID.

        :param replica: text -- ID of share replica
        """
    self.wait_for_resource_deletion(SHARE_REPLICA, res_id=replica, interval=3, timeout=60, microversion=microversion)