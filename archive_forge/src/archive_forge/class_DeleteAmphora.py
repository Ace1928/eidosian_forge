from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class DeleteAmphora(command.Command):
    """Delete a amphora"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('amphora_id', metavar='<amphora-id>', help='UUID of the amphora to delete.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        return parser

    def take_action(self, parsed_args):
        self.app.client_manager.load_balancer.amphora_delete(amphora_id=parsed_args.amphora_id)
        if parsed_args.wait:
            v2_utils.wait_for_delete(status_f=self.app.client_manager.load_balancer.amphora_show, res_id=parsed_args.amphora_id, status_field=const.STATUS)