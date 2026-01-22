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
def _verify_dict_keys(expected_keys, target_dict, strict=True):
    """Verify expected keys in a dictionary.

    :param expected_keys: A list of keys expected to be present.
    :param target_dict: The dictionary which should be verified.
    :param strict: Specifies whether additional keys are allowed to be present.
    :returns: None if the expected keys are found. Otherwise a human readable
        message indicating why the validation failed.
    """
    if not isinstance(target_dict, dict):
        msg_data = {'target_dict': target_dict, 'expected_keys': expected_keys}
        msg = "Invalid input. '%(target_dict)s' must be a dictionary with keys: %(expected_keys)s"
        LOG.debug(msg, msg_data)
        return _(msg) % msg_data
    expected_keys = set(expected_keys)
    provided_keys = set(target_dict.keys())
    predicate = expected_keys.__eq__ if strict else expected_keys.issubset
    if not predicate(provided_keys):
        msg_data = {'expected_keys': expected_keys, 'provided_keys': provided_keys}
        msg = "Validation of dictionary's keys failed. Expected keys: %(expected_keys)s Provided keys: %(provided_keys)s"
        LOG.debug(msg, msg_data)
        return _(msg) % msg_data