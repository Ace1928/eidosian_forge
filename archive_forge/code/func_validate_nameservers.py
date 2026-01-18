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
def validate_nameservers(data, valid_values=None):
    """Validate a list of unique IP addresses.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if data is a list of valid IP addresses, otherwise
        a human readable message is returned indicating why validation failed.
    """
    if not hasattr(data, '__iter__'):
        msg = "Invalid data format for nameserver: '%s'"
        LOG.debug(msg, data)
        return _(msg) % data
    hosts = []
    for host in data:
        msg = validate_ip_address(host)
        if msg:
            msg_data = {'host': host, 'msg': msg}
            msg = "'%(host)s' is not a valid nameserver. %(msg)s"
            LOG.debug(msg, msg_data)
            return _(msg) % msg_data
        if host in hosts:
            msg = "Duplicate nameserver '%s'"
            LOG.debug(msg, host)
            return _(msg) % host
        hosts.append(host)