import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.common import constants
class ListShareServer(command.Lister):
    """List all share servers (Admin only)."""
    _description = _('List all share servers (Admin only).')

    def get_parser(self, prog_name):
        parser = super(ListShareServer, self).get_parser(prog_name)
        parser.add_argument('--host', metavar='<hostname>', default=None, help=_('Filter results by name of host.'))
        parser.add_argument('--status', metavar='<status>', default=None, help=_('Filter results by status.'))
        parser.add_argument('--share-network', metavar='<share-network>', default=None, help=_('Filter results by share network name or ID.'))
        parser.add_argument('--project', metavar='<project>', default=None, help=_('Filter results by project name or ID.'))
        parser.add_argument('--share-network-subnet', metavar='<share-network-subnet>', type=str, default=None, help=_("Filter results by share network subnet that the share server's network allocation exists within. Available for microversion >= 2.51 (Optional, Default=None)"))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        identity_client = self.app.client_manager.identity
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        if parsed_args.share_network_subnet and share_client.api_version < api_versions.APIVersion('2.51'):
            raise exceptions.CommandError('Share network subnet can be specified only with manila API version >= 2.51')
        columns = ['ID', 'Host', 'Status', 'Share Network ID', 'Project ID']
        search_opts = {'status': parsed_args.status, 'host': parsed_args.host, 'project_id': project_id}
        if parsed_args.share_network:
            share_network_id = osc_utils.find_resource(share_client.share_networks, parsed_args.share_network).id
            search_opts['share_network'] = share_network_id
        if parsed_args.share_network_subnet:
            search_opts['share_network_subnet_id'] = parsed_args.share_network_subnet
        share_servers = share_client.share_servers.list(search_opts=search_opts)
        data = (osc_utils.get_dict_properties(share_server._info, columns) for share_server in share_servers)
        return (columns, data)