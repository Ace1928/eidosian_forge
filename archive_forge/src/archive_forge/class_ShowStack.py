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
class ShowStack(command.ShowOne):
    """Show stack details."""
    log = logging.getLogger(__name__ + '.ShowStack')

    def get_parser(self, prog_name):
        parser = super(ShowStack, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help='Stack to display (name or ID)')
        parser.add_argument('--no-resolve-outputs', action='store_true', help=_('Do not resolve outputs of the stack.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _show_stack(heat_client, stack_id=parsed_args.stack, format=parsed_args.formatter, resolve_outputs=not parsed_args.no_resolve_outputs)