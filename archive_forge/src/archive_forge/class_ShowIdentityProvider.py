import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowIdentityProvider(command.ShowOne):
    _description = _('Display identity provider details')

    def get_parser(self, prog_name):
        parser = super(ShowIdentityProvider, self).get_parser(prog_name)
        parser.add_argument('identity_provider', metavar='<identity-provider>', help=_('Identity provider to display'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        idp = utils.find_resource(identity_client.federation.identity_providers, parsed_args.identity_provider, id=parsed_args.identity_provider)
        idp._info.pop('links', None)
        remote_ids = format_columns.ListColumn(idp._info.pop('remote_ids', []))
        idp._info['remote_ids'] = remote_ids
        return zip(*sorted(idp._info.items()))