import logging
import netaddr
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class ListAddressGroup(command.Lister):
    _description = _('List address groups')

    def get_parser(self, prog_name):
        parser = super(ListAddressGroup, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help=_('List only address groups of given name in output'))
        parser.add_argument('--project', metavar='<project>', help=_('List address groups according to their project (name or ID)'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('id', 'name', 'description', 'project_id', 'addresses')
        column_headers = ('ID', 'Name', 'Description', 'Project', 'Addresses')
        attrs = {}
        if parsed_args.name:
            attrs['name'] = parsed_args.name
        if 'project' in parsed_args and parsed_args.project is not None:
            identity_client = self.app.client_manager.identity
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
            attrs['project_id'] = project_id
        data = client.address_groups(**attrs)
        return (column_headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))