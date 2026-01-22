from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class DeleteAgent(neutronV20.DeleteCommand):
    """Delete a given agent."""
    resource = 'agent'
    allow_names = False