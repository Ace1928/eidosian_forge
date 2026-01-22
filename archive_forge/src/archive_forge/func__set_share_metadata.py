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
def _set_share_metadata(self, share, data, update_all=False, microversion=None):
    """Sets a share metadata.

        :param share: str -- Name or ID of a share.
        :param data: dict -- key-value pairs to set as metadata.
        :param update_all: bool -- if set True then all keys except provided
            will be deleted.
        """
    if not (isinstance(data, dict) and data):
        msg = 'Provided invalid data for setting of share metadata - %s' % data
        raise exceptions.InvalidData(message=msg)
    if update_all:
        cmd = 'metadata-update-all %s ' % share
    else:
        cmd = 'metadata %s set ' % share
    for k, v in data.items():
        cmd += '%(k)s=%(v)s ' % {'k': k, 'v': v}
    return self.manila(cmd, microversion=microversion)