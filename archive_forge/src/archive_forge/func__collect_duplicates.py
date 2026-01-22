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
def _collect_duplicates(data_list):
    """Collects duplicate items from a list and returns them.

    :param data_list: A list of items to check for duplicates. The list may
        include dict items.
    :returns: A list of items that are duplicates in data_list. If no
        duplicates are found, the returned list is empty.
    """
    seen = []
    dups = []
    for datum in data_list:
        if datum in seen and datum not in dups:
            dups.append(datum)
            continue
        seen.append(datum)
    return dups