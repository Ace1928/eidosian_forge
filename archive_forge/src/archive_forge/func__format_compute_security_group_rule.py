import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def _format_compute_security_group_rule(sg_rule):
    info = network_utils.transform_compute_security_group_rule(sg_rule)
    info.pop('parent_group_id', None)
    keys_to_trim = ['ip_protocol', 'ip_range', 'port_range', 'remote_security_group']
    for key in keys_to_trim:
        if key in info and (not info[key]):
            info.pop(key)
    return utils.format_dict(info)