import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class UpdatePort(neutronV20.UpdateCommand, UpdatePortSecGroupMixin, UpdateExtraDhcpOptMixin, qos_policy.UpdateQosPolicyMixin, UpdatePortAllowedAddressPair):
    """Update port's information."""
    resource = 'port'

    def add_known_arguments(self, parser):
        _add_updatable_args(parser)
        parser.add_argument('--admin-state-up', choices=['True', 'False'], help=_('Set admin state up for the port.'))
        parser.add_argument('--admin_state_up', choices=['True', 'False'], help=argparse.SUPPRESS)
        self.add_arguments_secgroup(parser)
        self.add_arguments_extradhcpopt(parser)
        self.add_arguments_qos_policy(parser)
        self.add_arguments_allowedaddresspairs(parser)
        dns.add_dns_argument_update(parser, self.resource, 'name')

    def args2body(self, parsed_args):
        body = {}
        client = self.get_client()
        _updatable_args2body(parsed_args, body, client)
        if parsed_args.admin_state_up:
            body['admin_state_up'] = parsed_args.admin_state_up
        self.args2body_secgroup(parsed_args, body)
        self.args2body_extradhcpopt(parsed_args, body)
        self.args2body_qos_policy(parsed_args, body)
        self.args2body_allowedaddresspairs(parsed_args, body)
        dns.args2body_dns_update(parsed_args, body, 'name')
        return {'port': body}