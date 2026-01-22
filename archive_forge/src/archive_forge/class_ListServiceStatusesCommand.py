import logging
from osc_lib.command import command
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2 import utils as v2_utils
class ListServiceStatusesCommand(command.Lister):
    """List service statuses"""
    columns = ['id', 'hostname', 'service_name', 'status', 'stats', 'capabilities']

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--hostname', help='Hostname', required=False)
        parser.add_argument('--service_name', help='Service Name', required=False)
        parser.add_argument('--status', help='Status', required=False)
        common.add_all_common_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        cols = self.columns
        criterion = {}
        for i in ['hostname', 'service_name', 'status']:
            v = getattr(parsed_args, i)
            if v is not None:
                criterion[i] = v
        data = v2_utils.get_all(client.service_statuses.list, criterion=criterion)
        for i, s in enumerate(data):
            data[i] = _format_status(s)
        return (cols, (utils.get_item_properties(s, cols) for s in data))