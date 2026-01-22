from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import network
from neutronclient.neutron.v2_0 import router
class AddRouterToL3Agent(neutronV20.NeutronCommand):
    """Add a router to a L3 agent."""

    def get_parser(self, prog_name):
        parser = super(AddRouterToL3Agent, self).get_parser(prog_name)
        parser.add_argument('l3_agent', metavar='L3_AGENT', help=_('ID of the L3 agent.'))
        parser.add_argument('router', metavar='ROUTER', help=_('Router to add.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'router', parsed_args.router)
        neutron_client.add_router_to_l3_agent(parsed_args.l3_agent, {'router_id': _id})
        print(_('Added router %s to L3 agent') % parsed_args.router, file=self.app.stdout)