import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
class ListShareNetwork(command.Lister):
    _description = _('List share networks')

    def get_parser(self, prog_name):
        parser = super(ListShareNetwork, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<share-network>', help=_('Filter share networks by name'))
        parser.add_argument('--name~', metavar='<share-network-name-pattern>', help=_('Filter share networks by name-pattern. Available only for microversion >= 2.36.'))
        parser.add_argument('--description', metavar='<share-network-description>', help=_('Filter share networks by description. Available only for microversion >= 2.36'))
        parser.add_argument('--description~', metavar='<share-network-description-pattern>', help=_('Filter share networks by description-pattern. Available only for microversion >= 2.36.'))
        parser.add_argument('--all-projects', action='store_true', default=False, help=_('Include all projects (admin only)'))
        parser.add_argument('--project', metavar='<project>', help=_('Filter share networks by project (name or ID) (admin only)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--created-since', metavar='<yyyy-mm-dd>', help=_('Filter share networks by date they were created after. The date can be in the format yyyy-mm-dd.'))
        parser.add_argument('--created-before', metavar='<yyyy-mm-dd>', help=_('Filter share networks by date they were created before. The date can be in the format yyyy-mm-dd.'))
        parser.add_argument('--security-service', metavar='<security-service>', help=_('Filter share networks by the name or ID of a security service attached to the network.'))
        parser.add_argument('--neutron-net-id', metavar='<neutron-net-id>', help=_('Filter share networks by the ID of a neutron network.'))
        parser.add_argument('--neutron-subnet-id', metavar='<neutron-subnet-id>', help=_('Filter share networks by the ID of a neutron sub network.'))
        parser.add_argument('--network-type', metavar='<network-type>', help=_('Filter share networks by the type of network. Examples include "flat", "vlan", "vxlan", "geneve", etc.'))
        parser.add_argument('--segmentation-id', metavar='<segmentation-id>', help=_('Filter share networks by the segmentation ID of network. Relevant only for segmented networks such as "vlan", "vxlan", "geneve", etc.'))
        parser.add_argument('--cidr', metavar='<X.X.X.X/X>', help=_('Filter share networks by the CIDR of network.'))
        parser.add_argument('--ip-version', metavar='4/6', choices=['4', '6'], help=_('Filter share networks by the IP Version of the network, either 4 or 6.'))
        parser.add_argument('--detail', action='store_true', default=False, help=_('List share networks with details'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        identity_client = self.app.client_manager.identity
        columns = ['ID', 'Name']
        if parsed_args.detail:
            columns.extend(['Status', 'Created At', 'Updated At', 'Description'])
        project_id = None
        if parsed_args.project:
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
        all_tenants = bool(parsed_args.project) or parsed_args.all_projects
        if parsed_args.all_projects:
            columns.append('Project ID')
        search_opts = {'all_tenants': all_tenants, 'project_id': project_id, 'name': parsed_args.name, 'created_since': parsed_args.created_since, 'created_before': parsed_args.created_before, 'neutron_net_id': parsed_args.neutron_net_id, 'neutron_subnet_id': parsed_args.neutron_subnet_id, 'network_type': parsed_args.network_type, 'segmentation_id': parsed_args.segmentation_id, 'cidr': parsed_args.cidr, 'ip_version': parsed_args.ip_version, 'security_service': parsed_args.security_service}
        if share_client.api_version >= api_versions.APIVersion('2.36'):
            search_opts['name~'] = getattr(parsed_args, 'name~')
            search_opts['description~'] = getattr(parsed_args, 'description~')
            search_opts['description'] = parsed_args.description
        elif parsed_args.description or getattr(parsed_args, 'name~') or getattr(parsed_args, 'description~'):
            raise exceptions.CommandError('Pattern based filtering (name~, description~ and description) is only available with manila API version >= 2.36')
        data = share_client.share_networks.list(search_opts=search_opts)
        return (columns, (oscutils.get_item_properties(s, columns) for s in data))