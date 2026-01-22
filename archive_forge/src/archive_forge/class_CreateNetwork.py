import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class CreateNetwork(neutronV20.CreateCommand, qos_policy.CreateQosPolicyMixin):
    """Create a network for a given tenant."""
    resource = 'network'

    def add_known_arguments(self, parser):
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up to false.'))
        parser.add_argument('--admin_state_down', dest='admin_state', action='store_false', help=argparse.SUPPRESS)
        parser.add_argument('--shared', action='store_true', help=_('Set the network as shared.'), default=argparse.SUPPRESS)
        parser.add_argument('--provider:network_type', metavar='<network_type>', help=_('The physical mechanism by which the virtual network is implemented.'))
        parser.add_argument('--provider:physical_network', metavar='<physical_network_name>', help=_('Name of the physical network over which the virtual network is implemented.'))
        parser.add_argument('--provider:segmentation_id', metavar='<segmentation_id>', help=_('VLAN ID for VLAN networks or tunnel-id for GRE/VXLAN networks.'))
        utils.add_boolean_argument(parser, '--vlan-transparent', default=argparse.SUPPRESS, help=_('Create a VLAN transparent network.'))
        parser.add_argument('name', metavar='NAME', help=_('Name of the network to be created.'))
        parser.add_argument('--description', help=_('Description of network.'))
        self.add_arguments_qos_policy(parser)
        availability_zone.add_az_hint_argument(parser, self.resource)
        dns.add_dns_argument_create(parser, self.resource, 'domain')

    def args2body(self, parsed_args):
        body = {'admin_state_up': parsed_args.admin_state}
        args2body_common(body, parsed_args)
        neutronV20.update_dict(parsed_args, body, ['shared', 'tenant_id', 'vlan_transparent', 'provider:network_type', 'provider:physical_network', 'provider:segmentation_id', 'description'])
        self.args2body_qos_policy(parsed_args, body)
        availability_zone.args2body_az_hint(parsed_args, body)
        dns.args2body_dns_create(parsed_args, body, 'domain')
        return {'network': body}