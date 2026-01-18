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
def validate_oneline_not_empty_string(data, max_len=None):
    """Validate data is a non-empty string without newline character.

    :param data: The data to validate.
    :param max_len: An optional cap on the length of the string.
    :returns: None if the data is a string without newline character,
        otherwise a human readable message indicating why validation failed.
    """
    msg = validate_not_empty_string(data, max_len=max_len)
    if msg:
        return msg
    if len(data.splitlines()) > 1:
        msg = "Multi-line string is not allowed: '%s'"
        LOG.debug(msg, data)
        return _(msg) % data