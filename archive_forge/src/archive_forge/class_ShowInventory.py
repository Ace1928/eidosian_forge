import collections
import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib.i18n import _
from osc_lib import utils
from oslo_utils import excutils
from osc_placement.resources import common
from osc_placement import version
class ShowInventory(command.ShowOne):
    """Show the inventory for a given resource provider/class pair."""

    def get_parser(self, prog_name):
        parser = super(ShowInventory, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        parser.add_argument('resource_class', metavar='<resource_class>', help=RC_HELP)
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = PER_CLASS_URL.format(uuid=parsed_args.uuid, resource_class=parsed_args.resource_class)
        resource = http.request('GET', url).json()
        url = USAGES_BASE_URL.format(uuid=parsed_args.uuid)
        resources = http.request('GET', url).json()['usages']
        resource['used'] = resources[parsed_args.resource_class]
        fields = FIELDS + ('used',)
        return (fields, utils.get_dict_properties(resource, fields))