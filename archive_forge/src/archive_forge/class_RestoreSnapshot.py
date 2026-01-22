import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from heatclient.common import format_utils
from heatclient import exc as heat_exc
class RestoreSnapshot(command.Command):
    """Restore stack snapshot"""
    log = logging.getLogger(__name__ + '.RestoreSnapshot')

    def get_parser(self, prog_name):
        parser = super(RestoreSnapshot, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack containing the snapshot'))
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('ID of the snapshot to restore'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return self._restore_snapshot(heat_client, parsed_args)

    def _restore_snapshot(self, heat_client, parsed_args):
        fields = {'stack_id': parsed_args.stack, 'snapshot_id': parsed_args.snapshot}
        try:
            heat_client.stacks.restore(**fields)
        except heat_exc.HTTPNotFound:
            raise exc.CommandError(_('Stack %(stack)s or snapshot %(snapshot)s not found.') % {'stack': parsed_args.stack, 'snapshot': parsed_args.snapshot})