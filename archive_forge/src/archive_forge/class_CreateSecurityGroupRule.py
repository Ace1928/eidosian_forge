import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateSecurityGroupRule(neutronV20.CreateCommand):
    """Create a security group rule."""
    resource = 'security_group_rule'

    def add_known_arguments(self, parser):
        parser.add_argument('--description', help=_('Description of security group rule.'))
        parser.add_argument('security_group_id', metavar='SECURITY_GROUP', help=_('ID or name of the security group to which the rule is added.'))
        parser.add_argument('--direction', type=utils.convert_to_lowercase, default='ingress', choices=['ingress', 'egress'], help=_('Direction of traffic: ingress/egress.'))
        parser.add_argument('--ethertype', help=_('IPv4/IPv6'))
        parser.add_argument('--protocol', type=utils.convert_to_lowercase, help=_('Protocol of packet. Allowed values are [icmp, icmpv6, tcp, udp] and integer representations [0-255].'))
        parser.add_argument('--port-range-min', help=_('Starting port range. For ICMP it is type.'))
        parser.add_argument('--port_range_min', help=argparse.SUPPRESS)
        parser.add_argument('--port-range-max', help=_('Ending port range. For ICMP it is code.'))
        parser.add_argument('--port_range_max', help=argparse.SUPPRESS)
        parser.add_argument('--remote-ip-prefix', help=_('CIDR to match on.'))
        parser.add_argument('--remote_ip_prefix', help=argparse.SUPPRESS)
        parser.add_argument('--remote-group-id', metavar='REMOTE_GROUP', help=_('ID or name of the remote security group to which the rule is applied.'))
        parser.add_argument('--remote_group_id', help=argparse.SUPPRESS)

    def args2body(self, parsed_args):
        _security_group_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'security_group', parsed_args.security_group_id)
        body = {'security_group_id': _security_group_id, 'direction': parsed_args.direction, 'ethertype': parsed_args.ethertype or generate_default_ethertype(parsed_args.protocol)}
        neutronV20.update_dict(parsed_args, body, ['protocol', 'port_range_min', 'port_range_max', 'remote_ip_prefix', 'tenant_id', 'description'])
        if parsed_args.remote_group_id:
            _remote_group_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'security_group', parsed_args.remote_group_id)
            body['remote_group_id'] = _remote_group_id
        return {'security_group_rule': body}