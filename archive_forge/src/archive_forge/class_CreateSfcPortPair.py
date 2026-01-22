import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class CreateSfcPortPair(command.ShowOne):
    _description = _('Create a port pair')

    def get_parser(self, prog_name):
        parser = super(CreateSfcPortPair, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the port pair'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the port pair'))
        parser.add_argument('--service-function-parameters', metavar='correlation=<correlation-type>,weight=<weight>', action=parseractions.MultiKeyValueAction, optional_keys=['correlation', 'weight'], help=_('Dictionary of service function parameters. Currently, correlation=(None|mpls|nsh) and weight are supported. Weight is an integer that influences the selection of a port pair within a port pair group for a flow. The higher the weight, the more flows will hash to the port pair. The default weight is 1.'))
        parser.add_argument('--ingress', metavar='<ingress>', required=True, help=_('Ingress neutron port (name or ID)'))
        parser.add_argument('--egress', metavar='<egress>', required=True, help=_('Egress neutron port (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args)
        obj = client.create_sfc_port_pair(**attrs)
        display_columns, columns = utils.get_osc_show_columns_for_sdk_resource(obj, _attr_map_dict, ['location', 'tenant_id'])
        data = utils.get_dict_properties(obj, columns)
        return (display_columns, data)