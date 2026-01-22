import argparse
from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
from octaviaclient.osc.v2 import validate
class ShowListenerStats(command.ShowOne):
    """Shows the current statistics for a listener."""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('listener', metavar='<listener>', help='Name or UUID of the listener.')
        return parser

    def take_action(self, parsed_args):
        rows = const.LOAD_BALANCER_STATS_ROWS
        attrs = v2_utils.get_listener_attrs(self.app.client_manager, parsed_args)
        listener_id = attrs.pop('listener_id')
        data = self.app.client_manager.load_balancer.listener_stats_show(listener_id=listener_id)
        return (rows, utils.get_dict_properties(data['stats'], rows, formatters={}))