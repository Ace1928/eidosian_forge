import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ListZonesCommand(command.Lister):
    """List zones"""
    columns = ['id', 'name', 'type', 'serial', 'status', 'action']

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', help='Zone Name', required=False)
        parser.add_argument('--email', help='Zone Email', required=False)
        parser.add_argument('--type', help='Zone Type', choices=['PRIMARY', 'SECONDARY'], default=None, required=False)
        parser.add_argument('--ttl', help='Time To Live (Seconds)', required=False)
        parser.add_argument('--description', help='Description', required=False)
        parser.add_argument('--status', help='Zone Status', required=False)
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
        if parsed_args.ttl is not None:
            criterion['ttl'] = parsed_args.ttl
        if parsed_args.description is not None:
            criterion['description'] = parsed_args.description
        if parsed_args.email is not None:
            criterion['email'] = parsed_args.email
        if parsed_args.status is not None:
            criterion['status'] = parsed_args.status
        data = get_all(client.zones.list, criterion)
        cols = list(self.columns)
        if client.session.all_projects:
            cols.insert(1, 'project_id')
        return (cols, (utils.get_item_properties(s, cols) for s in data))