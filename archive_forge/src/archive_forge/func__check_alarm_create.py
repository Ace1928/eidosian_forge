from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.resources.openstack.aodh.gnocchi import alarm as gnocchi
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _check_alarm_create(self):
    self.fc.alarm.create.assert_called_once_with({'alarm_actions': [], 'description': 'Do stuff with gnocchi aggregation by resource', 'enabled': True, 'insufficient_data_actions': [], 'ok_actions': [], 'name': mock.ANY, 'type': 'gnocchi_aggregation_by_resources_threshold', 'repeat_actions': True, 'gnocchi_aggregation_by_resources_threshold_rule': {'aggregation_method': 'mean', 'granularity': 60, 'evaluation_periods': 1, 'threshold': 50, 'comparison_operator': 'gt', 'metric': 'cpu_util', 'resource_type': 'instance', 'query': '{"=": {"server_group": "my_autoscaling_group"}}'}, 'time_constraints': [], 'severity': 'low'})