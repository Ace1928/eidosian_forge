from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class ListAgent(neutronV20.ListCommand):
    """List agents."""
    resource = 'agent'
    list_columns = ['id', 'agent_type', 'host', 'availability_zone', 'alive', 'admin_state_up', 'binary']
    _formatters = {'heartbeat_timestamp': _format_timestamp}
    sorting_support = True

    def extend_list(self, data, parsed_args):
        for agent in data:
            if 'alive' in agent:
                agent['alive'] = ':-)' if agent['alive'] else 'xxx'