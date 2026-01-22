import argparse
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.vpn import utils as vpn_utils
class ShowIPsecPolicy(neutronv20.ShowCommand):
    """Show information of a given IPsec policy."""
    resource = 'ipsecpolicy'
    help_resource = 'IPsec policy'