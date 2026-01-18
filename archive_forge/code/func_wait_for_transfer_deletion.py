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
def wait_for_transfer_deletion(self, transfer, microversion=None):
    """Wait for transfer deletion by its Name or ID.

        :param transfer: str -- Name or ID of transfer.
        """
    self.wait_for_resource_deletion(SHARE, res_id=transfer, interval=5, timeout=300, microversion=microversion)