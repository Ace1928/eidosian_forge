import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class DeleteSharedZoneCommand(command.Command):
    """Delete a Zone Share"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('zone', help='The zone name or ID to share.')
        parser.add_argument('shared_zone_id', help='The zone share ID to delete.')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        client.zone_share.delete(parsed_args.zone, parsed_args.shared_zone_id)
        LOG.info('Shared Zone %s was deleted', parsed_args.shared_zone_id)