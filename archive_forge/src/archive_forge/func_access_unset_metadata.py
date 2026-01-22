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
def access_unset_metadata(self, access_id, keys, microversion=None):
    if not (isinstance(keys, (list, tuple, set)) and keys):
        raise exceptions.InvalidData(message='Provided invalid keys - %s' % keys)
    cmd = 'access-metadata %s unset ' % access_id
    for key in keys:
        cmd += '%s ' % key
    return self.manila(cmd, microversion=microversion)