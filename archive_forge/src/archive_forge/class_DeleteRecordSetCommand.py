import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class DeleteRecordSetCommand(command.ShowOne):
    """Delete recordset"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('zone_id', help='Zone ID')
        parser.add_argument('id', help='RecordSet ID')
        common.add_all_common_options(parser)
        common.add_edit_managed_option(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.recordsets.delete(parsed_args.zone_id, parsed_args.id)
        LOG.info('RecordSet %s was deleted', parsed_args.id)
        _format_recordset(data)
        return zip(*sorted(data.items()))