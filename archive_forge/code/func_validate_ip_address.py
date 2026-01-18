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
def validate_ip_address(data, valid_values=None):
    """Validate data is an IP address.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if data is an IP address, otherwise a human readable
        message indicating why data isn't an IP address.
    """
    msg = None
    msg_data = data
    try:
        ip = netaddr.IPAddress(validate_no_whitespace(data), flags=netaddr.core.ZEROFILL)
        if ':' not in data and data.count('.') != 3:
            msg = "'%s' is not a valid IP address"
        elif ip.version == 4 and str(ip) != data:
            msg_data = {'data': data, 'ip': ip}
            msg = "'%(data)s' is not an accepted IP address, '%(ip)s' is recommended"
    except Exception:
        msg = "'%s' is not a valid IP address"
    if msg:
        LOG.debug(msg, msg_data)
        msg = _(msg) % msg_data
    return msg