import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
from neutronclient.osc.v2 import utils as v2_utils
class CreateFirewallGroup(command.ShowOne):
    _description = _('Create a new firewall group')

    def get_parser(self, prog_name):
        parser = super(CreateFirewallGroup, self).get_parser(prog_name)
        _get_common_parser(parser)
        osc_utils.add_project_owner_option_to_parser(parser)
        port_group = parser.add_mutually_exclusive_group()
        port_group.add_argument('--port', metavar='<port>', action='append', help=_('Port(s) (name or ID) to apply firewall group.  This option can be repeated'))
        port_group.add_argument('--no-port', dest='no_port', action='store_true', help=_('Detach all port from the firewall group'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args)
        obj = client.create_firewall_group(**attrs)
        display_columns, columns = utils.get_osc_show_columns_for_sdk_resource(obj, _attr_map_dict, ['location', 'tenant_id'])
        data = utils.get_dict_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)