import collections
import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib.i18n import _
from osc_lib import utils
from oslo_utils import excutils
from osc_placement.resources import common
from osc_placement import version
class DeleteInventory(command.Command, version.CheckerMixin):
    """Delete the inventory.

    Depending on the resource class argument presence, delete all inventory for
    a given resource provider or for a resource provider/class pair.

    Delete all inventories for given resource provider requires at least
    ``--os-placement-api-version 1.5``.
    """

    def get_parser(self, prog_name):
        parser = super(DeleteInventory, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        parser.add_argument('--resource-class', metavar='<resource_class>', required=self.compare_version(version.lt('1.5')), help=RC_HELP + '\nThis argument can be omitted starting with ``--os-placement-api-version 1.5``. If it is omitted all inventories of the specified resource provider will be deleted.')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL
        params = {'uuid': parsed_args.uuid}
        if parsed_args.resource_class is not None:
            url = PER_CLASS_URL
            params = {'uuid': parsed_args.uuid, 'resource_class': parsed_args.resource_class}
        http.request('DELETE', url.format(**params))