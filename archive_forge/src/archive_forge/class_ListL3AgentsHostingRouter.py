from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import network
from neutronclient.neutron.v2_0 import router
class ListL3AgentsHostingRouter(neutronV20.ListCommand):
    """List L3 agents hosting a router."""
    resource = 'agent'
    _formatters = {}
    list_columns = ['id', 'host', 'admin_state_up', 'alive']
    unknown_parts_flag = False

    def get_parser(self, prog_name):
        parser = super(ListL3AgentsHostingRouter, self).get_parser(prog_name)
        parser.add_argument('router', metavar='ROUTER', help=_('Router to query.'))
        return parser

    def extend_list(self, data, parsed_args):
        if any(('ha_state' in agent for agent in data)):
            if 'ha_state' not in self.list_columns:
                self.list_columns.append('ha_state')
        for agent in data:
            agent['alive'] = ':-)' if agent['alive'] else 'xxx'

    def call_server(self, neutron_client, search_opts, parsed_args):
        _id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'router', parsed_args.router)
        search_opts['router'] = _id
        data = neutron_client.list_l3_agent_hosting_routers(**search_opts)
        return data