import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils as oscutils
class ListShareSnapshot(command.Lister):
    """List snapshots."""
    _description = _('List snapshots')

    def get_parser(self, prog_name):
        parser = super(ListShareSnapshot, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Display snapshots from all projects (Admin only).'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Filter results by name.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Filter results by description. Available only for microversion >= 2.36.'))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Filter results by status'))
        parser.add_argument('--share', metavar='<share>', default=None, help=_('Name or ID of a share to filter results by.'))
        parser.add_argument('--usage', metavar='<usage>', default=None, choices=['used', 'unused'], help=_('Option to filter snapshots by usage.'))
        parser.add_argument('--limit', metavar='<num-snapshots>', type=int, default=None, action=parseractions.NonNegativeAction, help=_('Limit the number of snapshots returned'))
        parser.add_argument('--marker', metavar='<snapshot>', help=_('The last share ID of the previous page'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', default='name:asc', help=_('Sort output by selected keys and directions(asc or desc) (default: name:asc), multiple keys and directions can be specified separated by comma'))
        parser.add_argument('--name~', metavar='<name~>', default=None, help=_('Filter results matching a share snapshot name pattern. Available only for microversion >= 2.36.'))
        parser.add_argument('--description~', metavar='<description~>', default=None, help=_('Filter results matching a share snapshot description pattern. Available only for microversion >= 2.36.'))
        parser.add_argument('--detail', action='store_true', default=False, help=_('List share snapshots with details'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Filter snapshots having a given metadata key=value property. (repeat option to filter by multiple properties)'))
        parser.add_argument('--count', action='store_true', default=False, help=_("The total count of share snapshots before pagination is applied. This parameter is useful when applying pagination parameters '--limit' and '--offset'. Available only for microversion >= 2.79."))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_id = None
        if parsed_args.share:
            share_id = utils.find_resource(share_client.shares, parsed_args.share).id
        columns = ['ID', 'Name']
        search_opts = {'offset': parsed_args.marker, 'limit': parsed_args.limit, 'all_tenants': parsed_args.all_projects, 'name': parsed_args.name, 'status': parsed_args.status, 'share_id': share_id, 'usage': parsed_args.usage, 'metadata': oscutils.extract_key_value_options(parsed_args.property)}
        if share_client.api_version >= api_versions.APIVersion('2.36'):
            search_opts['name~'] = getattr(parsed_args, 'name~')
            search_opts['description~'] = getattr(parsed_args, 'description~')
            search_opts['description'] = parsed_args.description
        elif parsed_args.description or getattr(parsed_args, 'name~') or getattr(parsed_args, 'description~'):
            raise exceptions.CommandError('Pattern based filtering (name~, description~ and description) is only available with manila API version >= 2.36')
        if parsed_args.count:
            if share_client.api_version < api_versions.APIVersion('2.79'):
                raise exceptions.CommandError('Displaying total number of share snapshots is only available with manila API version >= 2.79')
            if parsed_args.formatter != 'table':
                raise exceptions.CommandError("Count can only be printed when using '--format table'")
        if parsed_args.detail:
            columns.extend(['Status', 'Description', 'Created At', 'Size', 'Share ID', 'Share Proto', 'Share Size', 'User ID'])
        if parsed_args.all_projects:
            columns.append('Project ID')
        total_count = 0
        if parsed_args.count:
            search_opts['with_count'] = True
            snapshots, total_count = share_client.share_snapshots.list(search_opts=search_opts)
        else:
            snapshots = share_client.share_snapshots.list(search_opts=search_opts)
        snapshots = utils.sort_items(snapshots, parsed_args.sort, str)
        if parsed_args.count:
            print('Total number of snapshots: %s' % total_count)
        return (columns, (utils.get_item_properties(s, columns) for s in snapshots))