import argparse
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
class SetSecurityGroup(common.NetworkAndComputeCommand, common.NeutronCommandWithExtraArgs):
    _description = _('Set security group properties')

    def update_parser_common(self, parser):
        parser.add_argument('group', metavar='<group>', help=_('Security group to modify (name or ID)'))
        parser.add_argument('--name', metavar='<new-name>', help=_('New security group name'))
        parser.add_argument('--description', metavar='<description>', help=_('New security group description'))
        stateful_group = parser.add_mutually_exclusive_group()
        stateful_group.add_argument('--stateful', action='store_true', default=None, help=_('Security group is stateful (Default)'))
        stateful_group.add_argument('--stateless', action='store_true', default=None, help=_('Security group is stateless'))
        return parser

    def update_parser_network(self, parser):
        _tag.add_tag_option_to_parser_for_set(parser, _('security group'), enhance_help=self.enhance_help_neutron)
        return parser

    def take_action_network(self, client, parsed_args):
        obj = client.find_security_group(parsed_args.group, ignore_missing=False)
        attrs = {}
        if parsed_args.name is not None:
            attrs['name'] = parsed_args.name
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        if parsed_args.stateful:
            attrs['stateful'] = True
        if parsed_args.stateless:
            attrs['stateful'] = False
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        client.update_security_group(obj, **attrs)
        _tag.update_tags_for_set(client, obj, parsed_args)

    def take_action_compute(self, client, parsed_args):
        data = client.api.security_group_find(parsed_args.group)
        if parsed_args.name is not None:
            data['name'] = parsed_args.name
        if parsed_args.description is not None:
            data['description'] = parsed_args.description
        client.api.security_group_set(data, data['name'], data['description'])