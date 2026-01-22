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
def access_deny(self, share_id, access_id, microversion=None):
    return self.manila('access-deny %(share_id)s %(access_id)s' % {'share_id': share_id, 'access_id': access_id}, microversion=microversion)