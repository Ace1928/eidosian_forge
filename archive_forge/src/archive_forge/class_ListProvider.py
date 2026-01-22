from cliff import lister
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ListProvider(lister.Lister):
    """List all providers"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        columns = const.PROVIDER_COLUMNS
        data = self.app.client_manager.load_balancer.provider_list()
        return (columns, (utils.get_dict_properties(s, columns, formatters={}) for s in data['providers']))