import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class AcceptTransferRequestCommand(command.ShowOne):
    """Accept a Zone Transfer Request"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--transfer-id', help='Transfer ID', type=str, required=True)
        parser.add_argument('--key', help='Transfer Key', type=str, required=True)
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.zone_transfers.accept_request(parsed_args.transfer_id, parsed_args.key)
        return zip(*sorted(data.items()))