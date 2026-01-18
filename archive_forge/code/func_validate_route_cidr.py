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
def validate_route_cidr(data, valid_values=None):
    """Validate data is a proper CIDR string.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if data is valid CIDR. Otherwise a human
              readable message as to why data is invalid.
    """
    msg = None
    msg_data = data
    try:
        net = netaddr.IPNetwork(validate_no_whitespace(data))
        if '/' not in data or net.network != net.ip:
            msg_data = {'data': data, 'cidr': net.cidr}
            msg = "'%(data)s' is not a recognized CIDR, '%(cidr)s' is recommended"
        elif net.is_loopback():
            msg = "'%s' is not a routable CIDR"
        else:
            return
    except Exception:
        msg = "'%s' is not a valid CIDR"
    if msg:
        LOG.debug(msg, msg_data)
    return _(msg) % msg_data