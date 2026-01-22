from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class FailoverAmphora(command.Command):
    """Force failover an amphora"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('amphora_id', metavar='<amphora-id>', help='UUID of the amphora.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_amphora_attrs(self.app.client_manager, parsed_args)
        amp_id = attrs.pop('amphora_id')
        amphora = self.app.client_manager.load_balancer.amphora_show(amp_id)
        self.app.client_manager.load_balancer.amphora_failover(amphora_id=amp_id)
        if parsed_args.wait:
            lb_id = amphora.get('loadbalancer_id')
            if lb_id:
                v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.load_balancer_show, res_id=lb_id)
            else:
                v2_utils.wait_for_delete(status_f=self.app.client_manager.load_balancer.amphora_show, res_id=amp_id)