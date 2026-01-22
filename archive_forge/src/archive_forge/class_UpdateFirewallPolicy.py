import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class UpdateFirewallPolicy(neutronv20.UpdateCommand):
    """Update a given firewall policy."""
    resource = 'firewall_policy'

    def add_known_arguments(self, parser):
        add_common_args(parser)
        parser.add_argument('--name', help=_('Name for the firewall policy.'))
        utils.add_boolean_argument(parser, '--shared', help=_('Update the sharing status of the policy. (True means shared).'))
        utils.add_boolean_argument(parser, '--audited', help=_('Update the audit status of the policy. (True means auditing is enabled).'))

    def args2body(self, parsed_args):
        return parse_common_args(self.get_client(), parsed_args)