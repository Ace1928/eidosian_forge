import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
from neutronclient.osc.v2 import utils as v2_utils
class ListFirewallGroup(command.Lister):
    _description = _('List firewall groups')

    def get_parser(self, prog_name):
        parser = super(ListFirewallGroup, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.firewall_groups()
        headers, columns = column_util.get_column_definitions(_attr_map, long_listing=parsed_args.long)
        return (headers, (utils.get_dict_properties(s, columns, formatters=_formatters) for s in obj))