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
def validate_uuid_list(data, valid_values=None):
    """Validate data is a list of UUID like values.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if data is an iterable that contains valid UUID values,
        otherwise a message is returned indicating why validation failed.
    """
    return _validate_uuid_list(data, valid_values)