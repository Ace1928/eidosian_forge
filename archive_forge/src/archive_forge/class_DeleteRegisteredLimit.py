import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as common_utils
class DeleteRegisteredLimit(command.Command):
    _description = _('Delete a registered limit')

    def get_parser(self, prog_name):
        parser = super(DeleteRegisteredLimit, self).get_parser(prog_name)
        parser.add_argument('registered_limit_id', metavar='<registered-limit-id>', nargs='+', help=_('Registered limit to delete (ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for registered_limit_id in parsed_args.registered_limit_id:
            try:
                identity_client.registered_limits.delete(registered_limit_id)
            except Exception as e:
                errors += 1
                from pprint import pprint
                pprint(type(e))
                LOG.error(_("Failed to delete registered limit with ID '%(id)s': %(e)s"), {'id': registered_limit_id, 'e': e})
        if errors > 0:
            total = len(parsed_args.registered_limit_id)
            msg = _('%(errors)s of %(total)s registered limits failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)