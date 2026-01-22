import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import event_utils
from heatclient.common import format_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class CancelStack(StackActionBase):
    """Cancel current task for a stack.

    Supported tasks for cancellation:

    * update
    * create
    """
    log = logging.getLogger(__name__ + '.CancelStack')

    def get_parser(self, prog_name):
        parser = self._get_parser(prog_name, _('Stack(s) to cancel (name or ID)'), _('Wait for cancel to complete'))
        parser.add_argument('--no-rollback', action='store_true', help=_('Cancel without rollback'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        rows = []
        columns = ['ID', 'Stack Name', 'Stack Status', 'Creation Time', 'Updated Time']
        heat_client = self.app.client_manager.orchestration
        if parsed_args.no_rollback:
            action = heat_client.actions.cancel_without_rollback
            allowed_statuses = ['create_in_progress', 'update_in_progress']
        else:
            action = heat_client.actions.cancel_update
            allowed_statuses = ['update_in_progress']
        for stack in parsed_args.stack:
            try:
                data = heat_client.stacks.get(stack_id=stack)
            except heat_exc.HTTPNotFound:
                raise exc.CommandError('Stack not found: %s' % stack)
            status = getattr(data, 'stack_status').lower()
            if status in allowed_statuses:
                data = _stack_action(stack, parsed_args, heat_client, action)
                rows += [utils.get_dict_properties(data.to_dict(), columns)]
            else:
                err = _("Stack %(id)s with status '%(status)s' not in cancelable state") % {'id': stack, 'status': status}
                raise exc.CommandError(err)
        return (columns, rows)