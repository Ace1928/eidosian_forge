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
def validate_subnet_list(data, valid_values=None):
    """Validate data is a list of subnet dicts.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if data is a valid list of subnet dicts, otherwise a human
        readable message as to why the data is invalid.
    """
    return _validate_subnet_list(data, valid_values)