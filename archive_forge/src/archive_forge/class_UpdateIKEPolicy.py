import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class UpdateIKEPolicy(neutronv20.UpdateCommand):
    """Update a given IKE policy."""
    resource = 'ikepolicy'
    help_resource = 'IKE policy'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Updated name of the IKE policy.'))
        add_common_args(parser, is_create=False)

    def args2body(self, parsed_args):
        return {'ikepolicy': parse_common_args2body(parsed_args, body={})}