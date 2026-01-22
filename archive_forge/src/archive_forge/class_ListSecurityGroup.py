import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListSecurityGroup(neutronV20.ListCommand):
    """List security groups that belong to a given tenant."""
    resource = 'security_group'
    list_columns = ['id', 'name', 'security_group_rules']
    _formatters = {'security_group_rules': _format_sg_rules}
    pagination_support = True
    sorting_support = True