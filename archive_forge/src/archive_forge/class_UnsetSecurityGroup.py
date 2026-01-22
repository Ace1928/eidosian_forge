import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
class UnsetSecurityGroup(command.Command):
    _description = _('Unset security group properties')

    def get_parser(self, prog_name):
        parser = super(UnsetSecurityGroup, self).get_parser(prog_name)
        parser.add_argument('group', metavar='<group>', help=_('Security group to modify (name or ID)'))
        _tag.add_tag_option_to_parser_for_unset(parser, _('security group'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_security_group(parsed_args.group, ignore_missing=False)
        _tag.update_tags_for_unset(client, obj, parsed_args)