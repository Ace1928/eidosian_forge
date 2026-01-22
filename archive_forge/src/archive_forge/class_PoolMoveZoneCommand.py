import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class PoolMoveZoneCommand(command.Command):
    """Move a zone to another pool"""

    def get_parser(self, prog_name):
        parser = super(PoolMoveZoneCommand, self).get_parser(prog_name)
        parser.add_argument('zone_id', help='Zone ID')
        parser.add_argument('--pool-id', help='Pool ID')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = {}
        if parsed_args.pool_id:
            data['pool_id'] = parsed_args.pool_id
        client.zones.pool_move(parsed_args.zone_id, data)
        LOG.info('Scheduled move for zone %(zone_id)s', {'zone_id': parsed_args.zone_id})