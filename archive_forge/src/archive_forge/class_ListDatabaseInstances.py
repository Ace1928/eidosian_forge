import argparse
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient.osc.v1 import base
from troveclient import utils as trove_utils
class ListDatabaseInstances(command.Lister):
    _description = _('List database instances')
    columns = ['ID', 'Name', 'Datastore', 'Datastore Version', 'Status', 'Operating Status', 'Public', 'Addresses', 'Flavor ID', 'Size', 'Role']
    admin_columns = columns + ['Server ID', 'Tenant ID']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseInstances, self).get_parser(prog_name)
        parser.add_argument('--limit', dest='limit', metavar='<limit>', default=None, help=_('Limit the number of results displayed.'))
        parser.add_argument('--marker', dest='marker', metavar='<ID>', type=str, default=None, help=_('Begin displaying the results for IDs greater than thespecified marker. When used with ``--limit``, set this to the last ID displayed in the previous run.'))
        parser.add_argument('--include_clustered', '--include-clustered', dest='include_clustered', action='store_true', default=False, help=_('Include instances that are part of a cluster (default %(default)s).  --include-clustered may be deprecated in the future, retaining just --include_clustered.'))
        parser.add_argument('--all-projects', dest='all_projects', action='store_true', default=False, help=_('Include database instances of all projects (admin only)'))
        parser.add_argument('--project-id', help=_('Include database instances of a specific project (admin only)'))
        return parser

    def take_action(self, parsed_args):
        extra_params = {}
        if parsed_args.all_projects or parsed_args.project_id:
            db_instances = self.app.client_manager.database.mgmt_instances
            cols = self.admin_columns
            if parsed_args.project_id:
                extra_params['project_id'] = parsed_args.project_id
        else:
            db_instances = self.app.client_manager.database.instances
            cols = self.columns
        instances = db_instances.list(limit=parsed_args.limit, marker=parsed_args.marker, include_clustered=parsed_args.include_clustered, **extra_params)
        if instances:
            instances_info = get_instances_info(instances)
            instances = [osc_utils.get_dict_properties(info, cols) for info in instances_info]
        return (cols, instances)