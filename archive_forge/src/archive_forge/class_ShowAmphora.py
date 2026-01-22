from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ShowAmphora(command.ShowOne):
    """Show the details of a single amphora"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('amphora_id', metavar='<amphora-id>', help='UUID of the amphora.')
        return parser

    def take_action(self, parsed_args):
        data = self.app.client_manager.load_balancer.amphora_show(amphora_id=parsed_args.amphora_id)
        rows = const.AMPHORA_ROWS
        formatters = {'loadbalancers': v2_utils.format_list, 'amphorae': v2_utils.format_list}
        return (rows, utils.get_dict_properties(data, rows, formatters=formatters))