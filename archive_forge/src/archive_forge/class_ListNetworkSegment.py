import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class ListNetworkSegment(command.Lister):
    _description = _('List network segments')

    def get_parser(self, prog_name):
        parser = super(ListNetworkSegment, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        parser.add_argument('--network', metavar='<network>', help=_('List network segments that belong to this network (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        filters = {}
        if parsed_args.network:
            _network = network_client.find_network(parsed_args.network, ignore_missing=False)
            filters = {'network_id': _network.id}
        data = network_client.segments(**filters)
        headers = ('ID', 'Name', 'Network', 'Network Type', 'Segment')
        columns = ('id', 'name', 'network_id', 'network_type', 'segmentation_id')
        if parsed_args.long:
            headers = headers + ('Physical Network',)
            columns = columns + ('physical_network',)
        return (headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))