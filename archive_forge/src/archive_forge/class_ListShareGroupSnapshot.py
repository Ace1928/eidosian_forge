import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class ListShareGroupSnapshot(command.Lister):
    """List share group snapshots."""
    _description = _('List share group snapshots')

    def get_parser(self, prog_name):
        parser = super(ListShareGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Display information from all projects (Admin only).'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Filter results by name.'))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Filter results by status.'))
        parser.add_argument('--share-group', metavar='<share-group>', default=None, help=_('Filter results by share group name or ID.'))
        parser.add_argument('--limit', metavar='<limit>', type=int, default=None, action=parseractions.NonNegativeAction, help=_('Limit the number of share groups returned'))
        parser.add_argument('--marker', metavar='<marker>', help=_('The last share group snapshot ID of the previous page'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', default='name:asc', help=_('Sort output by selected keys and directions(asc or desc) (default: name:asc), multiple keys and directions can be specified separated by comma'))
        parser.add_argument('--detailed', action='store_true', help=_('Show detailed information about share group snapshot. '))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_group_id = None
        if parsed_args.share_group:
            share_group_id = osc_utils.find_resource(share_client.share_groups, parsed_args.share_group).id
        columns = ['ID', 'Name', 'Status', 'Description']
        search_opts = {'all_tenants': parsed_args.all_projects, 'name': parsed_args.name, 'status': parsed_args.status, 'share_group_id': share_group_id, 'limit': parsed_args.limit, 'offset': parsed_args.marker}
        if parsed_args.detailed:
            columns.extend(['Created At', 'Share Group ID'])
        if parsed_args.all_projects:
            columns.append('Project ID')
        share_group_snapshots = share_client.share_group_snapshots.list(search_opts=search_opts)
        share_group_snapshots = utils.sort_items(share_group_snapshots, parsed_args.sort, str)
        data = (osc_utils.get_dict_properties(share_group_snapshot._info, columns) for share_group_snapshot in share_group_snapshots)
        return (columns, data)