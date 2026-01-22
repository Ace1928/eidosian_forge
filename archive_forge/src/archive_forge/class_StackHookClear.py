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
class StackHookClear(command.Command):
    """Clear resource hooks on a given stack."""
    log = logging.getLogger(__name__ + '.StackHookClear')

    def get_parser(self, prog_name):
        parser = super(StackHookClear, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Stack to display (name or ID)'))
        parser.add_argument('--pre-create', action='store_true', help=_('Clear the pre-create hooks'))
        parser.add_argument('--pre-update', action='store_true', help=_('Clear the pre-update hooks'))
        parser.add_argument('--pre-delete', action='store_true', help=_('Clear the pre-delete hooks'))
        parser.add_argument('hook', metavar='<resource>', nargs='+', help=_('Resource names with hooks to clear. Resources in nested stacks can be set using slash as a separator: ``nested_stack/another/my_resource``. You can use wildcards to match multiple stacks or resources: ``nested_stack/an*/*_resource``'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _hook_clear(parsed_args, heat_client)