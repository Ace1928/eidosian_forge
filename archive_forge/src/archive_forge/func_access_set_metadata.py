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
def access_set_metadata(self, access_id, metadata, microversion=None):
    if not (isinstance(metadata, dict) and metadata):
        msg = 'Provided invalid metadata for setting of access rule metadata - %s' % metadata
        raise exceptions.InvalidData(message=msg)
    cmd = 'access-metadata %s set ' % access_id
    for k, v in metadata.items():
        cmd += '%(k)s=%(v)s ' % {'k': k, 'v': v}
    return self.manila(cmd, microversion=microversion)