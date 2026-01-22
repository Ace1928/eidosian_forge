import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class DeleteEndpoint(command.Command):
    _description = _('Delete endpoint(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteEndpoint, self).get_parser(prog_name)
        parser.add_argument('endpoints', metavar='<endpoint-id>', nargs='+', help=_('Endpoint(s) to delete (ID only)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        result = 0
        for endpoint in parsed_args.endpoints:
            try:
                identity_client.endpoints.delete(endpoint)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete endpoint with ID '%(endpoint)s': %(e)s"), {'endpoint': endpoint, 'e': e})
        if result > 0:
            total = len(parsed_args.endpoints)
            msg = _('%(result)s of %(total)s endpoints failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)