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
def unset_share_type_extra_specs(self, share_type_name_or_id, extra_specs_keys, microversion=None):
    """Unset key-value pair for share type."""
    if not (isinstance(extra_specs_keys, (list, tuple, set)) and extra_specs_keys):
        raise exceptions.InvalidData(message='Provided invalid extra specs - %s' % extra_specs_keys)
    cmd = 'type-key %s unset ' % share_type_name_or_id
    for key in extra_specs_keys:
        cmd += '%s ' % key
    return self.manila(cmd, microversion=microversion)