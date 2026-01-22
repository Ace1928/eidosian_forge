import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
class ListShareTransfer(command.Lister):
    """Lists all transfers."""
    _description = _('Lists all transfers')

    def get_parser(self, prog_name):
        parser = super(ListShareTransfer, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', help=_('Shows details for all tenants. (Admin only).'))
        parser.add_argument('--name', metavar='<name>', default=None, help='Filter share transfers by name. Default=None.')
        parser.add_argument('--id', metavar='<id>', default=None, help='Filter share transfers by ID. Default=None.')
        parser.add_argument('--resource-type', '--resource_type', metavar='<resource_type>', default=None, help='Filter share transfers by resource type, which can be share. Default=None.')
        parser.add_argument('--resource-id', '--resource_id', metavar='<resource_id>', default=None, help='Filter share transfers by resource ID. Default=None.')
        parser.add_argument('--source-project-id', '--source_project_id', metavar='<source_project_id>', default=None, help='Filter share transfers by ID of the Project that initiated the transfer. Default=None.')
        parser.add_argument('--limit', metavar='<limit>', type=int, default=None, help='Maximum number of transfer records to return. (Default=None)')
        parser.add_argument('--offset', metavar='<offset>', default=None, help='Start position of transfer records listing.')
        parser.add_argument('--sort-key', '--sort_key', metavar='<sort_key>', type=str, default=None, help='Key to be sorted, available keys are %(keys)s. Default=None.' % {'keys': constants.SHARE_TRANSFER_SORT_KEY_VALUES})
        parser.add_argument('--sort-dir', '--sort_dir', metavar='<sort_dir>', type=str, default=None, help='Sort direction, available values are %(values)s. OPTIONAL: Default=None.' % {'values': constants.SORT_DIR_VALUES})
        parser.add_argument('--detailed', dest='detailed', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Show detailed information about filtered share transfers.')
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        columns = ['ID', 'Name', 'Resource Type', 'Resource Id']
        if parsed_args.detailed:
            columns.extend(['Created At', 'Source Project Id', 'Destination Project Id', 'Accepted', 'Expires At'])
        search_opts = {'all_tenants': parsed_args.all_projects, 'id': parsed_args.id, 'name': parsed_args.name, 'limit': parsed_args.limit, 'offset': parsed_args.offset, 'resource_type': parsed_args.resource_type, 'resource_id': parsed_args.resource_id, 'source_project_id': parsed_args.source_project_id}
        transfers = share_client.transfers.list(detailed=parsed_args.detailed, search_opts=search_opts, sort_key=parsed_args.sort_key, sort_dir=parsed_args.sort_dir)
        return (columns, (osc_utils.get_item_properties(m, columns) for m in transfers))