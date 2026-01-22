import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteSecurityGroup(neutronV20.DeleteCommand):
    """Delete a given security group."""
    resource = 'security_group'
    allow_names = True