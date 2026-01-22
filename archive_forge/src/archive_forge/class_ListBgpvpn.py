import logging
from osc_lib.cli import format_columns
from osc_lib.cli.parseractions import KeyValueAction
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
class ListBgpvpn(command.Lister):
    _description = _('List BGP VPN resources')

    def get_parser(self, prog_name):
        parser = super(ListBgpvpn, self).get_parser(prog_name)
        nc_osc_utils.add_project_owner_option_to_parser(parser)
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output'))
        parser.add_argument('--property', metavar='<key=value>', default=dict(), help=_('Filter property to apply on returned BGP VPNs (repeat to filter on multiple properties)'), action=KeyValueAction)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        params = {}
        if parsed_args.project is not None:
            project_id = nc_osc_utils.find_project(self.app.client_manager.identity, parsed_args.project, parsed_args.project_domain).id
            params['tenant_id'] = project_id
        if parsed_args.property:
            params.update(parsed_args.property)
        objs = client.bgpvpns(**params)
        headers, columns = column_util.get_column_definitions(_attr_map, long_listing=parsed_args.long)
        return (headers, (osc_utils.get_dict_properties(s, columns, formatters=_formatters) for s in objs))