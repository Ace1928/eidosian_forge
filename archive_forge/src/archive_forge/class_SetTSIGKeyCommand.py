import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class SetTSIGKeyCommand(command.ShowOne):
    """Set tsigkey properties"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('id', help='TSIGKey ID')
        parser.add_argument('--name', help='TSIGKey Name')
        parser.add_argument('--algorithm', help='TSIGKey algorithm')
        parser.add_argument('--secret', help='TSIGKey secret')
        parser.add_argument('--scope', help='TSIGKey scope')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        data = {}
        if parsed_args.name:
            data['name'] = parsed_args.name
        if parsed_args.algorithm:
            data['algorithm'] = parsed_args.algorithm
        if parsed_args.secret:
            data['secret'] = parsed_args.secret
        if parsed_args.scope:
            data['scope'] = parsed_args.scope
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.tsigkeys.update(parsed_args.id, data)
        _format_tsigkey(data)
        return zip(*sorted(data.items()))