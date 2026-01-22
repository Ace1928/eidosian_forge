import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListConsistencyGroupSnapshot(command.Lister):
    _description = _('List consistency group snapshots.')

    def get_parser(self, prog_name):
        parser = super(ListConsistencyGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', help=_('Show detail for all projects (admin only) (defaults to False)'))
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output'))
        parser.add_argument('--status', metavar='<status>', choices=['available', 'error', 'creating', 'deleting', 'error_deleting'], help=_('Filters results by a status ("available", "error", "creating", "deleting" or "error_deleting")'))
        parser.add_argument('--consistency-group', metavar='<consistency-group>', help=_('Filters results by a consistency group (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.long:
            columns = ['ID', 'Status', 'ConsistencyGroup ID', 'Name', 'Description', 'Created At']
        else:
            columns = ['ID', 'Status', 'Name']
        volume_client = self.app.client_manager.volume
        consistency_group_id = None
        if parsed_args.consistency_group:
            consistency_group_id = utils.find_resource(volume_client.consistencygroups, parsed_args.consistency_group).id
        search_opts = {'all_tenants': parsed_args.all_projects, 'status': parsed_args.status, 'consistencygroup_id': consistency_group_id}
        consistency_group_snapshots = volume_client.cgsnapshots.list(detailed=True, search_opts=search_opts)
        return (columns, (utils.get_item_properties(s, columns) for s in consistency_group_snapshots))