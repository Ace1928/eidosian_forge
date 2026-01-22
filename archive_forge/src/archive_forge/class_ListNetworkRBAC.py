import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ListNetworkRBAC(command.Lister):
    _description = _('List network RBAC policies')

    def get_parser(self, prog_name):
        parser = super(ListNetworkRBAC, self).get_parser(prog_name)
        parser.add_argument('--type', metavar='<type>', choices=['address_group', 'address_scope', 'security_group', 'subnetpool', 'qos_policy', 'network'], help=_('List network RBAC policies according to given object type ("address_group", "address_scope", "security_group", "subnetpool", "qos_policy" or "network")'))
        parser.add_argument('--action', metavar='<action>', choices=['access_as_external', 'access_as_shared'], help=_('List network RBAC policies according to given action ("access_as_external" or "access_as_shared")'))
        parser.add_argument('--target-project', metavar='<target-project>', help=_('List network RBAC policies for a specific target project'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'object_type', 'object_id')
        column_headers = ('ID', 'Object Type', 'Object ID')
        query = {}
        if parsed_args.long:
            columns += ('action',)
            column_headers += ('Action',)
        if parsed_args.type is not None:
            query['object_type'] = parsed_args.type
        if parsed_args.action is not None:
            query['action'] = parsed_args.action
        if parsed_args.target_project is not None:
            project_id = '*'
            if parsed_args.target_project != '*':
                identity_client = self.app.client_manager.identity
                project_id = identity_common.find_project(identity_client, parsed_args.target_project).id
            query['target_project_id'] = project_id
        data = client.rbac_policies(**query)
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))