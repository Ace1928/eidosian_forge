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
@not_found_wrapper
def unset_share_metadata(self, share, keys, microversion=None):
    """Unsets some share metadata by keys.

        :param share: str -- Name or ID of a share
        :param keys: str/list -- key or list of keys to unset.
        """
    if not (isinstance(keys, list) and keys):
        msg = 'Provided invalid data for unsetting of share metadata - %s' % keys
        raise exceptions.InvalidData(message=msg)
    cmd = 'metadata %s unset ' % share
    for key in keys:
        cmd += '%s ' % key
    return self.manila(cmd, microversion=microversion)