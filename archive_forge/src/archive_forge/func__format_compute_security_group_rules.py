import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def _format_compute_security_group_rules(sg_rules):
    rules = []
    for sg_rule in sg_rules:
        rules.append(_format_compute_security_group_rule(sg_rule))
    return utils.format_list(rules, separator='\n')