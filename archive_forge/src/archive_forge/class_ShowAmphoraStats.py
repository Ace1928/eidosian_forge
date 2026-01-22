from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ShowAmphoraStats(command.ShowOne):
    """Shows the current statistics for an amphora."""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--listener', metavar='<listener>', help='Filter by listener (name or ID).')
        parser.add_argument('amphora_id', metavar='<amphora-id>', help='UUID of the amphora.')
        return parser

    def take_action(self, parsed_args):
        rows = const.LOAD_BALANCER_STATS_ROWS
        listener_id = None
        if parsed_args.listener is not None:
            attrs = v2_utils.get_listener_attrs(self.app.client_manager, parsed_args)
            listener_id = attrs.pop('listener_id')
        data = self.app.client_manager.load_balancer.amphora_stats_show(amphora_id=parsed_args.amphora_id)
        total_stats = {key: 0 for key in rows}
        for stats in data['amphora_stats']:
            if listener_id is None or listener_id == stats['listener_id']:
                for key in stats:
                    if key in rows:
                        total_stats[key] += stats[key]
        return (rows, utils.get_dict_properties(total_stats, rows, formatters={}))