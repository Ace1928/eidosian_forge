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
def validate_subports(data, valid_values=None):
    """Validate data is a list of subnet port dicts.

    :param data: The data to validate.
    :param valid_values: Not used!
    :returns: None if data is a list of subport dicts each with a unique valid
        port_id, segmentation_id and segmentation_type. Otherwise a human
        readable message is returned indicating why the data is invalid.
    """
    if not isinstance(data, list):
        msg = "Invalid data format for subports: '%s' is not a list"
        LOG.debug(msg, data)
        return _(msg) % data
    subport_ids = set()
    segmentations = collections.defaultdict(set)
    for subport in data:
        if not isinstance(subport, dict):
            msg = "Invalid data format for subport: '%s' is not a dict"
            LOG.debug(msg, subport)
            return _(msg) % subport
        if 'port_id' not in subport:
            msg = 'A valid port UUID must be specified'
            LOG.debug(msg)
            return _(msg)
        elif validate_uuid(subport['port_id']):
            msg = _("Invalid UUID for subport: '%s'") % subport['port_id']
            return msg
        elif subport['port_id'] in subport_ids:
            msg = _("Non unique UUID for subport: '%s'") % subport['port_id']
            return msg
        subport_ids.add(subport['port_id'])
        segmentation_type = subport.get('segmentation_type')
        if segmentation_type == 'inherit':
            return
        segmentation_id = subport.get('segmentation_id')
        if (not segmentation_type or segmentation_id is None) and len(subport) > 1:
            msg = "Invalid subport details '%s': missing segmentation information. Must specify both segmentation_id and segmentation_type"
            LOG.debug(msg, subport)
            return _(msg) % subport
        if segmentation_id in segmentations.get(segmentation_type, []):
            msg_data = {'seg_id': segmentation_id, 'subport': subport['port_id']}
            msg = "Segmentation ID '%(seg_id)s' for '%(subport)s' is not unique"
            LOG.debug(msg, msg_data)
            return _(msg) % msg_data
        if segmentation_id is not None:
            segmentations[segmentation_type].add(segmentation_id)