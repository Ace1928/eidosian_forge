import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class CreateSfcServiceGraph(command.ShowOne):
    """Create a service graph."""

    def get_parser(self, prog_name):
        parser = super(CreateSfcServiceGraph, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of the service graph.'))
        parser.add_argument('--description', help=_('Description for the service graph.'))
        parser.add_argument('--branching-point', metavar='SRC_CHAIN:DST_CHAIN_1,DST_CHAIN_2,DST_CHAIN_N', dest='branching_points', action='append', default=[], required=True, help=_('Service graph branching point: the key is the source Port Chain while the value is a list of destination Port Chains. This option can be repeated.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args)
        try:
            obj = client.create_sfc_service_graph(**attrs)
            display_columns, columns = utils.get_osc_show_columns_for_sdk_resource(obj, _attr_map_dict, ['location', 'tenant_id'])
            data = utils.get_dict_properties(obj, columns)
            return (display_columns, data)
        except Exception as e:
            msg = _("Failed to create service graph using '%(pcs)s': %(e)s") % {'pcs': parsed_args.branching_points, 'e': e}
            raise exceptions.CommandError(msg)