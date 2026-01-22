import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class RemoveRouterFromAgent(command.Command):
    _description = _('Remove router from an agent')

    def get_parser(self, prog_name):
        parser = super(RemoveRouterFromAgent, self).get_parser(prog_name)
        parser.add_argument('--l3', action='store_true', help=_('Remove router from an L3 agent'))
        parser.add_argument('agent_id', metavar='<agent-id>', help=_('Agent from which router will be removed (ID only)'))
        parser.add_argument('router', metavar='<router>', help=_('Router to be removed from an agent (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        agent = client.get_agent(parsed_args.agent_id)
        router = client.find_router(parsed_args.router, ignore_missing=False)
        if parsed_args.l3:
            client.remove_router_from_agent(agent, router)