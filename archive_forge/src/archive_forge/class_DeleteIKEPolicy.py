import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class DeleteIKEPolicy(neutronv20.DeleteCommand):
    """Delete a given IKE policy."""
    resource = 'ikepolicy'
    help_resource = 'IKE policy'