from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import network
from neutronclient.neutron.v2_0 import router
class ListRoutersOnL3Agent(neutronV20.ListCommand):
    """List the routers on a L3 agent."""
    _formatters = {'external_gateway_info': router._format_external_gateway_info}
    list_columns = ['id', 'name', 'external_gateway_info']
    resource = 'router'
    unknown_parts_flag = False

    def get_parser(self, prog_name):
        parser = super(ListRoutersOnL3Agent, self).get_parser(prog_name)
        parser.add_argument('l3_agent', metavar='L3_AGENT', help=_('ID of the L3 agent to query.'))
        return parser

    def call_server(self, neutron_client, search_opts, parsed_args):
        data = neutron_client.list_routers_on_l3_agent(parsed_args.l3_agent, **search_opts)
        return data