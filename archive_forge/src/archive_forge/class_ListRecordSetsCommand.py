import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ListRecordSetsCommand(command.Lister):
    """List recordsets"""
    columns = ['id', 'name', 'type', 'records', 'status', 'action']

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', help='RecordSet Name', required=False)
        parser.add_argument('--type', help='RecordSet Type', required=False)
        parser.add_argument('--data', help='RecordSet Record Data', required=False)
        parser.add_argument('--ttl', help='Time To Live (Seconds)', required=False)
        parser.add_argument('--description', help='Description', required=False)
        parser.add_argument('--status', help='RecordSet Status', required=False)
        parser.add_argument('--action', help='RecordSet Action', required=False)
        parser.add_argument('zone_id', help="Zone ID. To list all recordsets specify 'all'")
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        criterion = {}
        if parsed_args.type is not None:
            criterion['type'] = parsed_args.type
        if parsed_args.name is not None:
            criterion['name'] = parsed_args.name
        if parsed_args.data is not None:
            criterion['data'] = parsed_args.data
        if parsed_args.ttl is not None:
            criterion['ttl'] = parsed_args.ttl
        if parsed_args.description is not None:
            criterion['description'] = parsed_args.description
        if parsed_args.status is not None:
            criterion['status'] = parsed_args.status
        if parsed_args.action is not None:
            criterion['action'] = parsed_args.action
        cols = list(self.columns)
        if parsed_args.zone_id == 'all':
            data = get_all(client.recordsets.list_all_zones, criterion=criterion)
            cols.insert(2, 'zone_name')
        else:
            data = get_all(client.recordsets.list, args=[parsed_args.zone_id], criterion=criterion)
        if client.session.all_projects and _has_project_id(data):
            cols.insert(1, 'project_id')
        for i, rs in enumerate(data):
            data[i] = _format_recordset(rs)
        return (cols, (utils.get_item_properties(s, cols) for s in data))