from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class DeleteAvailabilityzone(command.Command):
    """Delete an availability zone"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('availabilityzone', metavar='<availabilityzone>', help='Name of the availability zone to delete.')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_availabilityzone_attrs(self.app.client_manager, parsed_args)
        availabilityzone_name = attrs.pop('availabilityzone_name')
        self.app.client_manager.load_balancer.availabilityzone_delete(availabilityzone_name=availabilityzone_name)