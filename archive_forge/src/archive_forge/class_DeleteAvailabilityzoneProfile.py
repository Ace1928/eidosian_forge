from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class DeleteAvailabilityzoneProfile(command.Command):
    """Delete an availability zone profile"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('availabilityzoneprofile', metavar='<availabilityzone_profile>', help='Availability zone profile to delete (name or ID).')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_availabilityzoneprofile_attrs(self.app.client_manager, parsed_args)
        availabilityzoneprofile_id = attrs.pop('availability_zone_profile_id')
        self.app.client_manager.load_balancer.availabilityzoneprofile_delete(availabilityzoneprofile_id=availabilityzoneprofile_id)