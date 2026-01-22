from cliff import lister
from cliff import show
from vitrageclient.common import utils
from vitrageclient import exceptions as exc
class AlarmShow(show.ShowOne):
    """Show an alarm"""

    def get_parser(self, prog_name):
        parser = super(AlarmShow, self).get_parser(prog_name)
        parser.add_argument('vitrage_id', help='Vitrage id of the alarm')
        return parser

    def take_action(self, parsed_args):
        vitrage_id = parsed_args.vitrage_id
        alarm = utils.get_client(self).alarm.get(vitrage_id=vitrage_id)
        return self.dict2columns(alarm)