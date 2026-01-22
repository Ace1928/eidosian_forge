import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
from openstackclient.network import utils as network_utils
class CreateDefaultSecurityGroupRule(command.ShowOne, common.NeutronCommandWithExtraArgs):
    """Add a new security group rule to the default security group template.

    These rules will be applied to the default security groups created for any
    new project. They will not be applied to any existing default security
    groups.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--description', metavar='<description>', help=_('Set default security group rule description'))
        parser.add_argument('--icmp-type', metavar='<icmp-type>', type=int, help=_('ICMP type for ICMP IP protocols'))
        parser.add_argument('--icmp-code', metavar='<icmp-code>', type=int, help=_('ICMP code for ICMP IP protocols'))
        direction_group = parser.add_mutually_exclusive_group()
        direction_group.add_argument('--ingress', action='store_true', help=_('Rule will apply to incoming network traffic (default)'))
        direction_group.add_argument('--egress', action='store_true', help=_('Rule will apply to outgoing network traffic'))
        parser.add_argument('--ethertype', metavar='<ethertype>', choices=['IPv4', 'IPv6'], type=network_utils.convert_ipvx_case, help=_('Ethertype of network traffic (IPv4, IPv6; default: based on IP protocol)'))
        remote_group = parser.add_mutually_exclusive_group()
        remote_group.add_argument('--remote-ip', metavar='<ip-address>', help=_('Remote IP address block (may use CIDR notation; default for IPv4 rule: 0.0.0.0/0, default for IPv6 rule: ::/0)'))
        remote_group.add_argument('--remote-group', metavar='<group>', help=_('Remote security group (ID)'))
        remote_group.add_argument('--remote-address-group', metavar='<group>', help=_('Remote address group (ID)'))
        parser.add_argument('--dst-port', metavar='<port-range>', action=parseractions.RangeAction, help=_('Destination port, may be a single port or a starting and ending port range: 137:139. Required for IP protocols TCP and UDP. Ignored for ICMP IP protocols.'))
        parser.add_argument('--protocol', metavar='<protocol>', type=network_utils.convert_to_lowercase, help=_('IP protocol (ah, dccp, egp, esp, gre, icmp, igmp, ipv66-encap, ipv6-frag, ipv6-icmp, ipv6-nonxt, ipv6-opts, ipv6-route, ospf, pgm, rsvp, sctp, tcp, udp, udplite, vrrp and integer representations [0-255] or any; default: any (all protocols))'))
        parser.add_argument('--for-default-sg', action='store_true', default=False, help=_('Set this default security group rule to be used in all default security groups created automatically for each project'))
        parser.add_argument('--for-custom-sg', action='store_true', default=True, help=_('Set this default security group rule to be used in all custom security groups created manually by users'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.sdk_connection.network
        attrs = {}
        attrs['protocol'] = network_utils.get_protocol(parsed_args)
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        if parsed_args.ingress or not parsed_args.egress:
            attrs['direction'] = 'ingress'
        if parsed_args.egress:
            attrs['direction'] = 'egress'
        attrs['ethertype'] = network_utils.get_ethertype(parsed_args, attrs['protocol'])
        if parsed_args.dst_port and (parsed_args.icmp_type or parsed_args.icmp_code):
            msg = _('Argument --dst-port not allowed with arguments --icmp-type and --icmp-code')
            raise exceptions.CommandError(msg)
        if parsed_args.icmp_type is None and parsed_args.icmp_code is not None:
            msg = _('Argument --icmp-type required with argument --icmp-code')
            raise exceptions.CommandError(msg)
        is_icmp_protocol = network_utils.is_icmp_protocol(attrs['protocol'])
        if not is_icmp_protocol and (parsed_args.icmp_type or parsed_args.icmp_code):
            msg = _('ICMP IP protocol required with arguments --icmp-type and --icmp-code')
            raise exceptions.CommandError(msg)
        if parsed_args.dst_port and (not is_icmp_protocol):
            attrs['port_range_min'] = parsed_args.dst_port[0]
            attrs['port_range_max'] = parsed_args.dst_port[1]
        if parsed_args.icmp_type is not None and parsed_args.icmp_type >= 0:
            attrs['port_range_min'] = parsed_args.icmp_type
        if parsed_args.icmp_code is not None and parsed_args.icmp_code >= 0:
            attrs['port_range_max'] = parsed_args.icmp_code
        if parsed_args.remote_group is not None:
            attrs['remote_group_id'] = parsed_args.remote_group
        elif parsed_args.remote_address_group is not None:
            attrs['remote_address_group_id'] = parsed_args.remote_address_group
        elif parsed_args.remote_ip is not None:
            attrs['remote_ip_prefix'] = parsed_args.remote_ip
        elif attrs['ethertype'] == 'IPv4':
            attrs['remote_ip_prefix'] = '0.0.0.0/0'
        elif attrs['ethertype'] == 'IPv6':
            attrs['remote_ip_prefix'] = '::/0'
        attrs['used_in_default_sg'] = parsed_args.for_default_sg
        attrs['used_in_non_default_sg'] = parsed_args.for_custom_sg
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_default_security_group_rule(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)