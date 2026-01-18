import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def update_parser_network(self, parser):
    _tag.add_tag_option_to_parser_for_set(parser, _('security group'), enhance_help=self.enhance_help_neutron)
    return parser