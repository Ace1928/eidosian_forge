import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class RemoveNetworkFromAgent(command.Command):
    _description = _('Remove network from an agent.')

    def get_parser(self, prog_name):
        parser = super(RemoveNetworkFromAgent, self).get_parser(prog_name)
        parser.add_argument('--dhcp', action='store_true', help=_('Remove network from DHCP agent'))
        parser.add_argument('agent_id', metavar='<agent-id>', help=_('Agent to which a network is removed (ID only)'))
        parser.add_argument('network', metavar='<network>', help=_('Network to be removed from an agent (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        agent = client.get_agent(parsed_args.agent_id)
        network = client.find_network(parsed_args.network, ignore_missing=False)
        if parsed_args.dhcp:
            try:
                client.remove_dhcp_agent_from_network(agent, network)
            except Exception:
                msg = 'Failed to remove {} to {}'.format(network.name, agent.agent_type)
                exceptions.CommandError(msg)