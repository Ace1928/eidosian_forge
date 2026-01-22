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
class ListStack(command.Lister):
    """List stacks."""
    log = logging.getLogger(__name__ + '.ListStack')

    def get_parser(self, prog_name):
        parser = super(ListStack, self).get_parser(prog_name)
        parser.add_argument('--deleted', action='store_true', help=_('Include soft-deleted stacks in the stack listing'))
        parser.add_argument('--nested', action='store_true', help=_('Include nested stacks in the stack listing'))
        parser.add_argument('--hidden', action='store_true', help=_('Include hidden stacks in the stack listing'))
        parser.add_argument('--property', dest='properties', metavar='<key=value>', help=_('Filter properties to apply on returned stacks (repeat to filter on multiple properties)'), action='append')
        parser.add_argument('--tags', metavar='<tag1,tag2...>', help=_('List of tags to filter by. Can be combined with --tag-mode to specify how to filter tags'))
        parser.add_argument('--tag-mode', metavar='<mode>', help=_('Method of filtering tags. Must be one of "any", "not", or "not-any". If not specified, multiple tags will be combined with the boolean AND expression'))
        parser.add_argument('--limit', metavar='<limit>', help=_('The number of stacks returned'))
        parser.add_argument('--marker', metavar='<id>', help=_('Only return stacks that appear after the given ID'))
        parser.add_argument('--sort', metavar='<key>[:<direction>]', help=_('Sort output by selected keys and directions (asc or desc) (default: asc). Specify multiple times to sort on multiple properties'))
        parser.add_argument('--all-projects', action='store_true', help=_('Include all projects (admin only)'))
        parser.add_argument('--short', action='store_true', help=_('List fewer fields in output'))
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output, this is implied by --all-projects'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        return _list(client, args=parsed_args)