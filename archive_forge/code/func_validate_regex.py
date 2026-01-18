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
def validate_regex(data, valid_values=None):
    """Validate data is matched against a regex.

    :param data: The data to validate.
    :param valid_values: The regular expression to use with re.match on
        the data.
    :returns: None if data contains matches for valid_values, otherwise a
        human readable message as to why data is invalid.
    """
    try:
        if re.match(valid_values, data):
            return
    except TypeError:
        pass
    msg = "'%s' is not a valid input"
    LOG.debug(msg, data)
    return _(msg) % data