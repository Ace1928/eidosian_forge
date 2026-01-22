import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from heatclient.common import format_utils
from heatclient import exc as heat_exc
class DeleteSnapshot(command.Command):
    """Delete stack snapshot."""
    log = logging.getLogger(__name__ + '.DeleteSnapshot')

    def get_parser(self, prog_name):
        parser = super(DeleteSnapshot, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack'))
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('ID of stack snapshot'))
        parser.add_argument('-y', '--yes', action='store_true', help=_('Skip yes/no prompt (assume yes)'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        msg = 'User did not confirm snapshot delete %sso taking no action.'
        try:
            if not parsed_args.yes and sys.stdin.isatty():
                sys.stdout.write(_('Are you sure you want to delete the snapshot of this stack [Y/N]?'))
                prompt_response = sys.stdin.readline().lower()
                if not prompt_response.startswith('y'):
                    self.log.info(msg, '')
                    return
        except KeyboardInterrupt:
            self.log.info(msg, '(ctrl-c) ')
            return
        except EOFError:
            self.log.info(msg, '(ctrl-d) ')
            return
        try:
            heat_client.stacks.snapshot_delete(parsed_args.stack, parsed_args.snapshot)
        except heat_exc.HTTPNotFound:
            raise exc.CommandError(_('Snapshot ID <%(snapshot_id)s> not found for stack <%(stack_id)s>') % {'snapshot_id': parsed_args.snapshot, 'stack_id': parsed_args.stack})