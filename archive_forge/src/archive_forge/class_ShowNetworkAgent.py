import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowNetworkAgent(command.ShowOne):
    _description = _('Display network agent details')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkAgent, self).get_parser(prog_name)
        parser.add_argument('network_agent', metavar='<network-agent>', help=_('Network agent to display (ID only)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.get_agent(parsed_args.network_agent)
        display_columns, columns = _get_network_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters=_formatters)
        return (display_columns, data)