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
def validate_fixed_ips(data, valid_values=None):
    """Validate data is a list of fixed IP dicts.

    In addition this function validates the ip_address and subnet_id
    if present in each fixed IP dict.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if data is a valid list of fixed IP dicts. Otherwise a
        human readable message is returned indicating why validation failed.
    """
    if not isinstance(data, list):
        msg = "Invalid data format for fixed IP: '%s'"
        LOG.debug(msg, data)
        return _(msg) % data
    ips = []
    for fixed_ip in data:
        if not isinstance(fixed_ip, dict):
            msg = "Invalid data format for fixed IP: '%s'"
            LOG.debug(msg, fixed_ip)
            return _(msg) % fixed_ip
        if 'ip_address' in fixed_ip:
            fixed_ip_address = fixed_ip['ip_address']
            if fixed_ip_address in ips:
                msg = "Duplicate IP address '%s'"
                LOG.debug(msg, fixed_ip_address)
                msg = _(msg) % fixed_ip_address
            else:
                msg = validate_ip_address(fixed_ip_address)
            if msg:
                return msg
            ips.append(fixed_ip_address)
        if 'subnet_id' in fixed_ip:
            msg = validate_uuid(fixed_ip['subnet_id'])
            if msg:
                return msg