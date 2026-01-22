from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class CreateAvailabilityzoneProfile(command.ShowOne):
    """Create an octavia availability zone profile"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', required=True, help='New octavia availability zone profile name.')
        parser.add_argument('--provider', metavar='<provider name>', required=True, help='Provider name for the availability zone profile.')
        parser.add_argument('--availability-zone-data', metavar='<availability_zone_data>', required=True, help='The JSON string containing the availability zone metadata.')
        return parser

    def take_action(self, parsed_args):
        rows = const.AVAILABILITYZONEPROFILE_ROWS
        attrs = v2_utils.get_availabilityzoneprofile_attrs(self.app.client_manager, parsed_args)
        body = {'availability_zone_profile': attrs}
        client_manager = self.app.client_manager
        data = client_manager.load_balancer.availabilityzoneprofile_create(json=body)
        return (rows, utils.get_dict_properties(data['availability_zone_profile'], rows, formatters={}))