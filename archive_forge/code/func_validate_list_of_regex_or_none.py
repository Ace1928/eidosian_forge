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
def validate_list_of_regex_or_none(data, valid_values=None):
    """Validate data is None or a list of items matching regex.

    :param data: A list of data to validate.
    :param valid_values: The regular expression to use with re.match on
        each element of the data.
    :returns: None if data is None or contains matches for valid_values,
        otherwise a human readable message as to why data is invalid.
    """
    if data is not None:
        return _validate_list_of_items(validate_regex, data, valid_values)