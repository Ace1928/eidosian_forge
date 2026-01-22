import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class CreateSfcPortPairGroup(command.ShowOne):
    _description = _('Create a port pair group')

    def get_parser(self, prog_name):
        parser = super(CreateSfcPortPairGroup, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the port pair group'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the port pair group'))
        parser.add_argument('--port-pair', metavar='<port-pair>', dest='port_pairs', default=[], action='append', help=_('Port pair (name or ID). This option can be repeated.'))
        tap_enable = parser.add_mutually_exclusive_group()
        tap_enable.add_argument('--enable-tap', action='store_true', help=_('Port pairs of this port pair group are deployed as passive tap service function'))
        tap_enable.add_argument('--disable-tap', action='store_true', help=_('Port pairs of this port pair group are deployed as l3 service function (default)'))
        parser.add_argument('--port-pair-group-parameters', metavar='lb-fields=<lb-fields>', action=parseractions.KeyValueAction, help=_('Dictionary of port pair group parameters. Currently only one parameter lb-fields is supported. <lb-fields> is a & separated list of load-balancing fields.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args)
        obj = client.create_sfc_port_pair_group(**attrs)
        display_columns, columns = utils.get_osc_show_columns_for_sdk_resource(obj, _attr_map_dict, ['location', 'tenant_id'])
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)