import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ShowFloatingIPCommand(command.ShowOne):
    """Show floatingip ptr record details"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('floatingip_id', help='Floating IP ID in format region:floatingip_id')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.floatingips.get(parsed_args.floatingip_id)
        _format_floatingip(data)
        return zip(*sorted(data.items()))