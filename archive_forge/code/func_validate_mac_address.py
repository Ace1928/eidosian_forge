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
def validate_mac_address(data, valid_values=None):
    """Validate data is a MAC address.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if the data is a valid MAC address, otherwise a human
        readable message as to why validation failed.
    """
    try:
        valid_mac = netaddr.valid_mac(validate_no_whitespace(data))
    except Exception:
        valid_mac = False
    if valid_mac:
        valid_mac = netaddr.EUI(data) not in map(netaddr.EUI, constants.INVALID_MAC_ADDRESSES)
    if not valid_mac:
        msg = "'%s' is not a valid MAC address"
        LOG.debug(msg, data)
        return _(msg) % data