import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class ListShareGroup(command.Lister):
    """List share groups."""
    _description = _('List share groups')

    def get_parser(self, prog_name):
        parser = super(ListShareGroup, self).get_parser(prog_name)
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Display share groups from all projects (Admin only).'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Filter results by name.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Filter results by description. Available only for microversion >= 2.36.'))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Filter results by status.'))
        parser.add_argument('--share-server', metavar='<share-server-id>', default=None, help=_('Filter results by share server ID.'))
        parser.add_argument('--share-group-type', metavar='<share-group-type>', default=None, help=_('Filter results by a share group type ID or name that was used for share group creation. '))
        parser.add_argument('--snapshot', metavar='<snapshot>', default=None, help=_('Filter results by share group snapshot name or ID that was used to create the share group. '))
        parser.add_argument('--host', metavar='<host>', default=None, help=_('Filter results by host.'))
        parser.add_argument('--share-network', metavar='<share-network>', default=None, help=_('Filter results by share-network name or ID. '))
        parser.add_argument('--project', metavar='<project>', default=None, help=_("Filter results by project name or ID. Useful with set key '--all-projects'. "))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--limit', metavar='<limit>', type=int, default=None, action=parseractions.NonNegativeAction, help=_('Limit the number of share groups returned'))
        parser.add_argument('--marker', metavar='<marker>', help=_('The last share group ID of the previous page'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', default='name:asc', help=_('Sort output by selected keys and directions(asc or desc) (default: name:asc), multiple keys and directions can be specified separated by comma'))
        parser.add_argument('--name~', metavar='<name~>', default=None, help=_('Filter results matching a share group name pattern. Available only for microversion >= 2.36. '))
        parser.add_argument('--description~', metavar='<description~>', default=None, help=_('Filter results matching a share group description pattern. Available only for microversion >= 2.36. '))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        identity_client = self.app.client_manager.identity
        share_server_id = None
        if parsed_args.share_server:
            share_server_id = osc_utils.find_resource(share_client.share_servers, parsed_args.share_server).id
        share_group_type = None
        if parsed_args.share_group_type:
            share_group_type = osc_utils.find_resource(share_client.share_group_types, parsed_args.share_group_type).id
        snapshot = None
        if parsed_args.snapshot:
            snapshot = apiutils.find_resource(share_client.share_snapshots, parsed_args.snapshot).id
        share_network = None
        if parsed_args.share_network:
            share_network = osc_utils.find_resource(share_client.share_networks, parsed_args.share_network).id
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        columns = ['ID', 'Name', 'Status', 'Description']
        search_opts = {'all_tenants': parsed_args.all_projects, 'name': parsed_args.name, 'status': parsed_args.status, 'share_server_id': share_server_id, 'share_group_type': share_group_type, 'snapshot': snapshot, 'host': parsed_args.host, 'share_network': share_network, 'project_id': project_id, 'limit': parsed_args.limit, 'offset': parsed_args.marker}
        if share_client.api_version >= api_versions.APIVersion('2.36'):
            search_opts['name~'] = getattr(parsed_args, 'name~')
            search_opts['description~'] = getattr(parsed_args, 'description~')
            search_opts['description'] = parsed_args.description
        elif parsed_args.description or getattr(parsed_args, 'name~') or getattr(parsed_args, 'description~'):
            raise exceptions.CommandError('Pattern based filtering (name~, description~ and description) is only available with manila API version >= 2.36')
        if parsed_args.all_projects:
            columns.append('Project ID')
        share_groups = share_client.share_groups.list(search_opts=search_opts)
        data = (osc_utils.get_dict_properties(share_group._info, columns) for share_group in share_groups)
        return (columns, data)