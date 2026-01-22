import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class SetBlacklistCommand(command.ShowOne):
    """Set blacklist properties"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('id', help='Blacklist ID')
        parser.add_argument('--pattern', help='Blacklist pattern')
        description_group = parser.add_mutually_exclusive_group()
        description_group.add_argument('--description', help='Description')
        description_group.add_argument('--no-description', action='store_true')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        data = {}
        if parsed_args.pattern:
            data['pattern'] = parsed_args.pattern
        if parsed_args.no_description:
            data['description'] = None
        elif parsed_args.description:
            data['description'] = parsed_args.description
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        updated = client.blacklists.update(parsed_args.id, data)
        _format_blacklist(updated)
        return zip(*sorted(updated.items()))