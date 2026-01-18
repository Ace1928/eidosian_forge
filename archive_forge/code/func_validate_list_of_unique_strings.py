import collections
import functools
import inspect
import re
import netaddr
from os_ken.lib.packet import ether_types
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from webob import exc
from neutron_lib._i18n import _
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.plugins import directory
from neutron_lib.services.qos import constants as qos_consts
def validate_list_of_unique_strings(data, max_len=None):
    """Validate data is a list of unique strings.

    :param data: The data to validate.
    :param max_len: An optional cap on the length of the string.
    :returns: None if the data is a list of non-empty/non-blank strings,
        otherwise a human readable message indicating why validation failed.
    """
    return _validate_list_of_unique_strings(data, max_len=max_len)