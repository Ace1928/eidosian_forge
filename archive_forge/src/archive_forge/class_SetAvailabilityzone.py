from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class SetAvailabilityzone(command.Command):
    """Update an availability zone"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('availabilityzone', metavar='<availabilityzone>', help='Name of the availability zone to update.')
        parser.add_argument('--description', metavar='<description>', help='Set the description of the availability zone.')
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Enable the availability zone.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Disable the availability zone.')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_availabilityzone_attrs(self.app.client_manager, parsed_args)
        availabilityzone_name = attrs.pop('availabilityzone_name')
        body = {'availability_zone': attrs}
        self.app.client_manager.load_balancer.availabilityzone_set(availabilityzone_name, json=body)