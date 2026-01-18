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
def validate_mac_address_or_none(data, valid_values=None):
    """Validate data is a MAC address if the data isn't None.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if the data is None or a valid MAC address, otherwise
        a human readable message indicating why validation failed.
    """
    if data is not None:
        return validate_mac_address(data, valid_values)