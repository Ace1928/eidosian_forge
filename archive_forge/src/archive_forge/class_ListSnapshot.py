import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
from heatclient.common import format_utils
from heatclient import exc as heat_exc
class ListSnapshot(command.Lister):
    """List stack snapshots."""
    log = logging.getLogger(__name__ + '.ListSnapshot')

    def get_parser(self, prog_name):
        parser = super(ListSnapshot, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack containing the snapshots'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return self._list_snapshot(heat_client, parsed_args)

    def _list_snapshot(self, heat_client, parsed_args):
        fields = {'stack_id': parsed_args.stack}
        try:
            snapshots = heat_client.stacks.snapshot_list(**fields)
        except heat_exc.HTTPNotFound:
            raise exc.CommandError(_('Stack not found: %s') % parsed_args.stack)
        columns = ['id', 'name', 'status', 'status_reason', 'creation_time']
        return (columns, (utils.get_dict_properties(s, columns) for s in snapshots['snapshots']))