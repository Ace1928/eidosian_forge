import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from heatclient.common import format_utils
from heatclient import exc as heat_exc
class ShowSnapshot(format_utils.YamlFormat):
    """Show stack snapshot."""
    log = logging.getLogger(__name__ + '.ShowSnapshot')

    def get_parser(self, prog_name):
        parser = super(ShowSnapshot, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack containing the snapshot'))
        parser.add_argument('snapshot', metavar='<snapshot>', help=_('ID of the snapshot to show'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return self._show_snapshot(heat_client, parsed_args.stack, parsed_args.snapshot)

    def _show_snapshot(self, heat_client, stack_id, snapshot_id):
        try:
            data = heat_client.stacks.snapshot_show(stack_id, snapshot_id)
        except heat_exc.HTTPNotFound:
            raise exc.CommandError(_('Snapshot ID <%(snapshot_id)s> not found for stack <%(stack_id)s>') % {'snapshot_id': snapshot_id, 'stack_id': stack_id})
        rows = list(data.values())
        columns = list(data.keys())
        return (columns, rows)