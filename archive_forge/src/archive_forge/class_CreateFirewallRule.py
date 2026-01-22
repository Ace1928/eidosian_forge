import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class CreateFirewallRule(neutronv20.CreateCommand):
    """Create a firewall rule."""
    resource = 'firewall_rule'

    def add_known_arguments(self, parser):
        parser.add_argument('--shared', action='store_true', help=_('Set shared flag for the firewall rule.'), default=argparse.SUPPRESS)
        _add_common_args(parser)
        parser.add_argument('--ip-version', type=int, choices=[4, 6], default=4, help=_('IP version for the firewall rule (default is 4).'))

    def args2body(self, parsed_args):
        return {self.resource: common_args2body(parsed_args)}