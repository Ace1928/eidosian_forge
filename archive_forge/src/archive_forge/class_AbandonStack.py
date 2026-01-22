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
class AbandonStack(format_utils.JsonFormat):
    """Abandon stack and output results."""
    log = logging.getLogger(__name__ + '.AbandonStack')

    def get_parser(self, prog_name):
        parser = super(AbandonStack, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack to abandon'))
        parser.add_argument('--output-file', metavar='<output-file>', help=_('File to output abandon results'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        try:
            stack = client.stacks.abandon(stack_id=parsed_args.stack)
        except heat_exc.HTTPNotFound:
            msg = _('Stack not found: %s') % parsed_args.stack
            raise exc.CommandError(msg)
        if parsed_args.output_file is not None:
            try:
                with open(parsed_args.output_file, 'w') as f:
                    f.write(jsonutils.dumps(stack, indent=2))
                    return ([], None)
            except IOError as e:
                raise exc.CommandError(str(e))
        data = list(stack.values())
        columns = list(stack.keys())
        return (columns, data)