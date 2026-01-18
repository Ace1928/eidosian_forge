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
def validate_boolean(data, valid_values=None):
    """Validate data is a python bool compatible object.

    :param data: The data to validate.
    :param valid_values: Not used!
    :return: None if the value can be converted to a bool, otherwise a
        human readable message indicating why data is invalid.
    """
    try:
        strutils.bool_from_string(data, strict=True)
    except ValueError:
        msg = "'%s' is not a valid boolean value"
        LOG.debug(msg, data)
        return _(msg) % data