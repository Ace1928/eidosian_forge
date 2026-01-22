import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class SetLocalIP(command.Command):
    _description = _('Set local ip properties')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('local_ip', metavar='<local-ip>', help=_('Local IP to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set local IP name'))
        parser.add_argument('--description', metavar='<description>', help=_('Set local IP description'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_local_ip(parsed_args.local_ip, ignore_missing=False)
        attrs = {}
        if parsed_args.name is not None:
            attrs['name'] = parsed_args.name
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        if attrs:
            client.update_local_ip(obj, **attrs)