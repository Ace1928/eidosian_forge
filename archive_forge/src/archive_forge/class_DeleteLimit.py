import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as common_utils
class DeleteLimit(command.Command):
    _description = _('Delete a limit')

    def get_parser(self, prog_name):
        parser = super(DeleteLimit, self).get_parser(prog_name)
        parser.add_argument('limit_id', metavar='<limit-id>', nargs='+', help=_('Limit to delete (ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for limit_id in parsed_args.limit_id:
            try:
                identity_client.limits.delete(limit_id)
            except Exception as e:
                errors += 1
                LOG.error(_("Failed to delete limit with ID '%(id)s': %(e)s"), {'id': limit_id, 'e': e})
        if errors > 0:
            total = len(parsed_args.limit_id)
            msg = _('%(errors)s of %(total)s limits failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)