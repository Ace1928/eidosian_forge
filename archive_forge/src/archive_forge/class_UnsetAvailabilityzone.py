from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class UnsetAvailabilityzone(command.Command):
    """Clear availability zone settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('availabilityzone', metavar='<availabilityzone>', help='Name of the availability zone to update.')
        parser.add_argument('--description', action='store_true', help='Clear the availability zone description.')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args:
            return
        availabilityzone_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.availabilityzone_list, 'availability_zones', parsed_args.availabilityzone)
        body = {'availability_zone': unset_args}
        self.app.client_manager.load_balancer.availabilityzone_set(availabilityzone_id, json=body)