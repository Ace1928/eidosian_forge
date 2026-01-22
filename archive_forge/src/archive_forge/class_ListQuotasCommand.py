import itertools
import logging
from cliff import command
from cliff import show
from designateclient.v2.cli import common
class ListQuotasCommand(show.ShowOne):
    """List quotas"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        common.add_all_common_options(parser)
        parser.add_argument('--project-id', help='Project ID Default: current project')
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.dns
        common.set_all_common_headers(client, parsed_args)
        proj_id = client.session.get_project_id()
        if parsed_args.project_id and parsed_args.project_id != proj_id:
            proj_id = parsed_args.project_id
            common.set_all_projects(client, True)
        data = client.quotas.list(proj_id)
        return zip(*sorted(data.items()))