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
def validate_hostroutes(data, valid_values=None):
    """Validate a list of unique host route dicts.

    :param data: The data to validate. To be valid it must be a list like
        structure of host route dicts, each containing 'destination' and
        'nexthop' key values.
    :param valid_values: Not used!
    :returns: None if data is a valid list of unique host route dicts,
        otherwise a human readable message indicating why validation failed.
    """
    if not isinstance(data, list):
        msg = "Invalid data format for hostroute: '%s'"
        LOG.debug(msg, data)
        return _(msg) % data
    expected_keys = ['destination', 'nexthop']
    hostroutes = []
    for hostroute in data:
        msg = _verify_dict_keys(expected_keys, hostroute)
        if msg:
            return msg
        msg = validate_route_cidr(hostroute['destination'])
        if msg:
            return msg
        msg = validate_ip_address(hostroute['nexthop'])
        if msg:
            return msg
        if hostroute in hostroutes:
            msg = "Duplicate hostroute '%s'"
            LOG.debug(msg, hostroute)
            return _(msg) % hostroute
        hostroutes.append(hostroute)