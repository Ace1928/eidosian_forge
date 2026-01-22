import collections
import ipaddress
from oslo_utils import uuidutils
import re
import string
from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient.v2 import share_instances
@staticmethod
def _validate_common_name(access):
    if len(access) == 0 or len(access) > 64:
        exc_str = 'Invalid CN (common name). Must be 1-64 chars long.'
        raise exceptions.CommandError(exc_str)