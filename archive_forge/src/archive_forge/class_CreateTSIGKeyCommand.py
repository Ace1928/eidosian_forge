import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class CreateTSIGKeyCommand(command.ShowOne):
    """Create new tsigkey"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', help='TSIGKey Name', required=True)
        parser.add_argument('--algorithm', help='TSIGKey algorithm', required=True)
        parser.add_argument('--secret', help='TSIGKey secret', required=True)
        parser.add_argument('--scope', help='TSIGKey scope', required=True)
        parser.add_argument('--resource-id', help='TSIGKey resource_id', required=True)
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.tsigkeys.create(parsed_args.name, parsed_args.algorithm, parsed_args.secret, parsed_args.scope, parsed_args.resource_id)
        _format_tsigkey(data)
        return zip(*sorted(data.items()))