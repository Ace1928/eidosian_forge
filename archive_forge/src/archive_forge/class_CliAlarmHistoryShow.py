from cliff import lister
from oslo_serialization import jsonutils
from aodhclient import utils
class CliAlarmHistoryShow(lister.Lister):
    """Show history for an alarm"""
    COLS = ('timestamp', 'type', 'detail', 'event_id')

    def get_parser(self, prog_name):
        parser = super(CliAlarmHistoryShow, self).get_parser(prog_name)
        parser.add_argument('alarm_id', metavar='<alarm-id>', help='ID of an alarm')
        parser.add_argument('--limit', type=int, metavar='<LIMIT>', help='Number of resources to return (Default is server default)')
        parser.add_argument('--marker', metavar='<MARKER>', help='Last item of the previous listing. Return the next results after this value,the supported marker is event_id.')
        parser.add_argument('--sort', action='append', metavar='<SORT_KEY:SORT_DIR>', help='Sort of resource attribute. e.g. timestamp:desc')
        return parser

    def take_action(self, parsed_args):
        history = utils.get_client(self).alarm_history.get(alarm_id=parsed_args.alarm_id, sorts=parsed_args.sort, limit=parsed_args.limit, marker=parsed_args.marker)
        return utils.list2cols(self.COLS, history)