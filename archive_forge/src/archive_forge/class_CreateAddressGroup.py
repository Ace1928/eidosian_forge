import logging
import netaddr
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class CreateAddressGroup(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create a new Address Group')

    def get_parser(self, prog_name):
        parser = super(CreateAddressGroup, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('New address group name'))
        parser.add_argument('--description', metavar='<description>', help=_('New address group description'))
        parser.add_argument('--address', metavar='<ip-address>', action='append', default=[], help=_('IP address or CIDR (repeat option to set multiple addresses)'))
        parser.add_argument('--project', metavar='<project>', help=_("Owner's project (name or ID)"))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_address_group(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)