import argparse
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from aodhclient import exceptions
from aodhclient.i18n import _
from aodhclient import utils
class CliAlarmStateSet(show.ShowOne):
    """Set state of an alarm"""

    def get_parser(self, prog_name):
        parser = _add_name_to_parser(_add_id_to_parser(super(CliAlarmStateSet, self).get_parser(prog_name)))
        parser.add_argument('--state', metavar='<STATE>', required=True, choices=ALARM_STATES, help='State of the alarm, one of: ' + str(ALARM_STATES))
        return parser

    def take_action(self, parsed_args):
        _check_name_and_id(parsed_args, 'set state of')
        c = utils.get_client(self)
        if parsed_args.name:
            _id = _find_alarm_id_by_name(c, parsed_args.name)
        elif uuidutils.is_uuid_like(parsed_args.id):
            try:
                state = c.alarm.set_state(parsed_args.id, parsed_args.state)
            except exceptions.NotFound:
                _id = _find_alarm_id_by_name(c, parsed_args.id)
            else:
                return self.dict2columns({'state': state})
        else:
            _id = _find_alarm_id_by_name(c, parsed_args.id)
        state = c.alarm.set_state(_id, parsed_args.state)
        return self.dict2columns({'state': state})