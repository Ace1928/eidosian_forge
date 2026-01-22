from cliff import lister
from osc_lib.command import command
from osc_lib import utils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ListAmphora(lister.Lister):
    """List amphorae"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--loadbalancer', metavar='<loadbalancer>', dest='loadbalancer', help='Filter by load balancer (name or ID).')
        parser.add_argument('--compute-id', metavar='<compute-id>', help='Filter by compute ID.')
        role_choices = {'MASTER', 'BACKUP', 'STANDALONE'}
        parser.add_argument('--role', metavar='{' + ','.join(sorted(role_choices)) + '}', choices=role_choices, type=lambda s: s.upper(), help='Filter by role.')
        status_choices = {'ALLOCATED', 'BOOTING', 'DELETED', 'ERROR', 'PENDING_CREATE', 'PENDING_DELETE', 'READY'}
        parser.add_argument('--status', '--provisioning-status', dest='status', metavar='{' + ','.join(sorted(status_choices)) + '}', choices=status_choices, type=lambda s: s.upper(), help='Filter by amphora provisioning status.')
        parser.add_argument('--long', action='store_true', help='Show additional fields.')
        return parser

    def take_action(self, parsed_args):
        columns = const.AMPHORA_COLUMNS
        if parsed_args.long:
            columns = const.AMPHORA_COLUMNS_LONG
        attrs = v2_utils.get_amphora_attrs(self.app.client_manager, parsed_args)
        data = self.app.client_manager.load_balancer.amphora_list(**attrs)
        formatters = {'amphorae': v2_utils.format_list}
        return (columns, (utils.get_dict_properties(amp, columns, formatters=formatters) for amp in data['amphorae']))