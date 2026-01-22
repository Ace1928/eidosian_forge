from stevedore import extension
from neutronclient.neutron import v2_0 as neutronV20
class NeutronClientExtension(neutronV20.NeutronCommand):
    pagination_support = False
    _formatters = {}
    sorting_support = False