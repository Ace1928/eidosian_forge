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
class CliAlarmUpdate(CliAlarmCreate):
    """Update an alarm"""
    create = False

    def get_parser(self, prog_name):
        return _add_id_to_parser(super(CliAlarmUpdate, self).get_parser(prog_name))

    def take_action(self, parsed_args):
        attributes = self._alarm_from_args(parsed_args)
        _check_name_and_id_exist(parsed_args, 'update')
        c = utils.get_client(self)
        if uuidutils.is_uuid_like(parsed_args.id):
            try:
                alarm = c.alarm.update(alarm_id=parsed_args.id, alarm_update=attributes)
            except exceptions.NotFound:
                _id = _find_alarm_id_by_name(c, parsed_args.id)
            else:
                return self.dict2columns(_format_alarm(alarm))
        elif parsed_args.id:
            _id = _find_alarm_id_by_name(c, parsed_args.id)
        else:
            _id = _find_alarm_id_by_name(c, parsed_args.name)
        alarm = c.alarm.update(alarm_id=_id, alarm_update=attributes)
        return self.dict2columns(_format_alarm(alarm))