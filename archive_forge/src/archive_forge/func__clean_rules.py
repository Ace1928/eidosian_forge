from oslo_serialization import jsonutils
from aodhclient import utils
from aodhclient.v2 import alarm_cli
from aodhclient.v2 import base
@staticmethod
def _clean_rules(alarm_type, alarm):
    for rule in alarm_cli.ALARM_TYPES:
        if rule != alarm_type:
            alarm.pop('%s_rule' % rule, None)