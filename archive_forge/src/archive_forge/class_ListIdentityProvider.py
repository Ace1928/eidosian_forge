import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ListIdentityProvider(command.Lister):
    _description = _('List identity providers')

    def get_parser(self, prog_name):
        parser = super(ListIdentityProvider, self).get_parser(prog_name)
        parser.add_argument('--id', metavar='<id>', help=_('The Identity Providers’ ID attribute'))
        parser.add_argument('--enabled', dest='enabled', action='store_true', help=_('The Identity Providers that are enabled will be returned'))
        return parser

    def take_action(self, parsed_args):
        columns = ('ID', 'Enabled', 'Domain ID', 'Description')
        identity_client = self.app.client_manager.identity
        kwargs = {}
        if parsed_args.id:
            kwargs['id'] = parsed_args.id
        if parsed_args.enabled:
            kwargs['enabled'] = True
        data = identity_client.federation.identity_providers.list(**kwargs)
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))