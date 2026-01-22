import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class CreateRecordSetCommand(command.ShowOne):
    """Create new recordset"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('zone_id', help='Zone ID')
        parser.add_argument('name', help='RecordSet Name')
        req_group = parser.add_mutually_exclusive_group(required=True)
        req_group.add_argument('--record', help='RecordSet Record, repeat if necessary', action='append')
        parser.add_argument('--type', help='RecordSet Type', required=True)
        parser.add_argument('--ttl', type=int, help='Time To Live (Seconds)')
        parser.add_argument('--description', help='Description')
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        data = client.recordsets.create(parsed_args.zone_id, parsed_args.name, parsed_args.type, parsed_args.record, description=parsed_args.description, ttl=parsed_args.ttl)
        _format_recordset(data)
        return zip(*sorted(data.items()))