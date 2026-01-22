import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class DeleteIdentityProvider(command.Command):
    _description = _('Delete identity provider(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteIdentityProvider, self).get_parser(prog_name)
        parser.add_argument('identity_provider', metavar='<identity-provider>', nargs='+', help=_('Identity provider(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        result = 0
        for i in parsed_args.identity_provider:
            try:
                identity_client.federation.identity_providers.delete(i)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete identity providers with name or ID '%(provider)s': %(e)s"), {'provider': i, 'e': e})
        if result > 0:
            total = len(parsed_args.identity_provider)
            msg = _('%(result)s of %(total)s identity providers failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)