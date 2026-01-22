import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ExportZoneCommand(command.ShowOne):
    """Export a Zone"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        common.add_all_common_options(parser)
        parser.add_argument('zone_id', help='Zone ID', type=str)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.zone_exports.create(parsed_args.zone_id)
        _format_zone_export_record(data)
        LOG.info('Zone Export %s was created', data['id'])
        return zip(*sorted(data.items()))