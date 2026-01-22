import json
import argparse
from cliff import command
from datetime import datetime
from iso8601 import iso8601
from iso8601 import ParseError
class EventPost(command.Command):
    """Post an event to Vitrage"""

    @staticmethod
    def iso8601(argument_value):
        try:
            if argument_value:
                iso8601.parse_date(argument_value)
        except ParseError:
            msg = '%s must be an iso8601 date' % argument_value
            raise argparse.ArgumentTypeError(msg)

    def get_parser(self, prog_name):
        parser = super(EventPost, self).get_parser(prog_name)
        parser.add_argument('--type', required=True, help='The type of the event')
        parser.add_argument('--time', default='', type=self.iso8601, help='The timestamp of the event in ISO 8601 format: YYYY-MM-DDTHH:MM:SS.mmmmmm. If not specified, the current time is used')
        parser.add_argument('--details', default='{}', help='A json string with the event details')
        return parser

    def take_action(self, parsed_args):
        if parsed_args.time:
            event_time = parsed_args.time
        else:
            event_time = datetime.now().isoformat()
        event_type = parsed_args.type
        details = parsed_args.details
        self.app.client.event.post(event_time=event_time, event_type=event_type, details=json.loads(details))