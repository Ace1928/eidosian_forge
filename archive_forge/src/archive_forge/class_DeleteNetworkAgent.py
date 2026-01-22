import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteNetworkAgent(command.Command):
    _description = _('Delete network agent(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkAgent, self).get_parser(prog_name)
        parser.add_argument('network_agent', metavar='<network-agent>', nargs='+', help=_('Network agent(s) to delete (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for agent in parsed_args.network_agent:
            try:
                client.delete_agent(agent, ignore_missing=False)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete network agent with ID '%(agent)s': %(e)s"), {'agent': agent, 'e': e})
        if result > 0:
            total = len(parsed_args.network_agent)
            msg = _('%(result)s of %(total)s network agents failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)