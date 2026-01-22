import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def _format_network_security_group_rules(sg_rules):
    for sg_rule in sg_rules:
        empty_keys = [k for k, v in sg_rule.items() if not v]
        for key in empty_keys:
            sg_rule.pop(key)
        sg_rule.pop('security_group_id', None)
        sg_rule.pop('tenant_id', None)
        sg_rule.pop('project_id', None)
    return utils.format_list_of_dicts(sg_rules)