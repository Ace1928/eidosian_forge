import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class UpdateNetwork(neutronV20.UpdateCommand, qos_policy.UpdateQosPolicyMixin):
    """Update network's information."""
    resource = 'network'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Name of the network.'))
        parser.add_argument('--description', help=_('Description of this network.'))
        self.add_arguments_qos_policy(parser)
        dns.add_dns_argument_update(parser, self.resource, 'domain')

    def args2body(self, parsed_args):
        body = {}
        args2body_common(body, parsed_args)
        self.args2body_qos_policy(parsed_args, body)
        dns.args2body_dns_update(parsed_args, body, 'domain')
        return {'network': body}