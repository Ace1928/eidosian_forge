import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class CreateFloatingIPPortForwarding(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create floating IP port forwarding')

    def get_parser(self, prog_name):
        parser = super(CreateFloatingIPPortForwarding, self).get_parser(prog_name)
        parser.add_argument('--internal-ip-address', required=True, metavar='<internal-ip-address>', help=_('The fixed IPv4 address of the network port associated to the floating IP port forwarding'))
        parser.add_argument('--port', metavar='<port>', required=True, help=_('The name or ID of the network port associated to the floating IP port forwarding'))
        parser.add_argument('--internal-protocol-port', metavar='<port-number>', required=True, help=_('The protocol port number of the network port fixed IPv4 address associated to the floating IP port forwarding'))
        parser.add_argument('--external-protocol-port', metavar='<port-number>', required=True, help=_("The protocol port number of the port forwarding's floating IP address"))
        (parser.add_argument('--protocol', metavar='<protocol>', required=True, help=_('The protocol used in the floating IP port forwarding, for instance: TCP, UDP')),)
        parser.add_argument('--description', metavar='<description>', help=_('A text to describe/contextualize the use of the port forwarding configuration'))
        parser.add_argument('floating_ip', metavar='<floating-ip>', help=_('Floating IP that the port forwarding belongs to (IP address or ID)'))
        return parser

    def take_action(self, parsed_args):
        attrs = {}
        client = self.app.client_manager.network
        floating_ip = client.find_ip(parsed_args.floating_ip, ignore_missing=False)
        validate_and_assign_port_ranges(parsed_args, attrs)
        if parsed_args.port:
            port = client.find_port(parsed_args.port, ignore_missing=False)
            attrs['internal_port_id'] = port.id
        attrs['internal_ip_address'] = parsed_args.internal_ip_address
        attrs['protocol'] = parsed_args.protocol
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_floating_ip_port_forwarding(floating_ip.id, **attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)