import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class CreateNetworkSegment(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create new network segment')

    def get_parser(self, prog_name):
        parser = super(CreateNetworkSegment, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('New network segment name'))
        parser.add_argument('--description', metavar='<description>', help=_('Network segment description'))
        parser.add_argument('--physical-network', metavar='<physical-network>', help=_('Physical network name of this network segment'))
        parser.add_argument('--segment', metavar='<segment>', type=int, help=_('Segment identifier for this network segment which is based on the network type, VLAN ID for vlan network type and tunnel ID for geneve, gre and vxlan network types'))
        parser.add_argument('--network', metavar='<network>', required=True, help=_('Network this network segment belongs to (name or ID)'))
        parser.add_argument('--network-type', metavar='<network-type>', choices=['flat', 'geneve', 'gre', 'local', 'vlan', 'vxlan'], required=True, help=_('Network type of this network segment (flat, geneve, gre, local, vlan or vxlan)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = {}
        attrs['name'] = parsed_args.name
        attrs['network_id'] = client.find_network(parsed_args.network, ignore_missing=False).id
        attrs['network_type'] = parsed_args.network_type
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        if parsed_args.physical_network is not None:
            attrs['physical_network'] = parsed_args.physical_network
        if parsed_args.segment is not None:
            attrs['segmentation_id'] = parsed_args.segment
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_segment(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)