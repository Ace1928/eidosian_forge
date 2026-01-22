import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class CreateBlacklistCommand(command.ShowOne):
    """Create new blacklist"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--pattern', help='Blacklist pattern', required=True)
        parser.add_argument('--description', help='Description')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.blacklists.create(parsed_args.pattern, parsed_args.description)
        _format_blacklist(data)
        return zip(*sorted(data.items()))