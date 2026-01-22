import logging
from osc_lib.command import command
from osc_lib import exceptions as osc_exc
from designateclient import utils
from designateclient.v2.cli import common
from designateclient.v2.utils import get_all
class ListSharedZonesCommand(command.Lister):
    """List Zone Shares"""
    columns = ['id', 'zone_id', 'target_project_id']

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        common.add_all_common_options(parser)
        parser.add_argument('zone', help='The zone name or ID to share.')
        parser.add_argument('--target-project-id', help='The target project ID to filter on.', required=False)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        criterion = {}
        if parsed_args.target_project_id is not None:
            criterion['target_project_id'] = parsed_args.target_project_id
        data = get_all(client.zone_share.list, criterion=criterion, args=[parsed_args.zone])
        cols = list(self.columns)
        if client.session.all_projects:
            cols.insert(1, 'project_id')
        return (cols, (utils.get_item_properties(s, cols) for s in data))