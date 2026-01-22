import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetConntrackHelper(command.Command):
    _description = _('Set L3 conntrack helper properties')

    def get_parser(self, prog_name):
        parser = super(SetConntrackHelper, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router that the conntrack helper belong to'))
        parser.add_argument('conntrack_helper_id', metavar='<conntrack-helper-id>', help=_('The ID of the conntrack helper(s)'))
        parser.add_argument('--helper', metavar='<helper>', help=_('The netfilter conntrack helper module'))
        parser.add_argument('--protocol', metavar='<protocol>', help=_('The network protocol for the netfilter conntrack target rule'))
        parser.add_argument('--port', metavar='<port>', type=int, help=_('The network port for the netfilter conntrack target rule'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(client, parsed_args)
        if attrs:
            client.update_conntrack_helper(parsed_args.conntrack_helper_id, attrs.pop('router_id'), **attrs)