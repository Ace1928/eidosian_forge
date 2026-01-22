from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import network
from neutronclient.neutron.v2_0 import router
class RemoveNetworkFromDhcpAgent(neutronV20.NeutronCommand):
    """Remove a network from a DHCP agent."""

    def get_parser(self, prog_name):
        parser = super(RemoveNetworkFromDhcpAgent, self).get_parser(prog_name)
        parser.add_argument('dhcp_agent', metavar='DHCP_AGENT', help=_('ID of the DHCP agent.'))
        parser.add_argument('network', metavar='NETWORK', help=_('Network to remove.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _net_id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'network', parsed_args.network)
        neutron_client.remove_network_from_dhcp_agent(parsed_args.dhcp_agent, _net_id)
        print(_('Removed network %s from DHCP agent') % parsed_args.network, file=self.app.stdout)