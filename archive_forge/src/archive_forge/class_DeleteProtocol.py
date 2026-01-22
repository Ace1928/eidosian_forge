import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteProtocol(command.Command):
    _description = _('Delete federation protocol(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteProtocol, self).get_parser(prog_name)
        parser.add_argument('federation_protocol', metavar='<federation-protocol>', nargs='+', help=_('Federation protocol(s) to delete (name or ID)'))
        parser.add_argument('--identity-provider', metavar='<identity-provider>', required=True, help=_('Identity provider that supports <federation-protocol> (name or ID) (required)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        result = 0
        for i in parsed_args.federation_protocol:
            try:
                identity_client.federation.protocols.delete(parsed_args.identity_provider, i)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete federation protocol with name or ID '%(protocol)s': %(e)s"), {'protocol': i, 'e': e})
        if result > 0:
            total = len(parsed_args.federation_protocol)
            msg = _('%(result)s of %(total)s federation protocols failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)