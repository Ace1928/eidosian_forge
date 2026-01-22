from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.resources.openstack.aodh.gnocchi import alarm as gnocchi
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class GnocchiAggregationByResourcesAlarmTest(GnocchiResourcesAlarmTest):

    def _check_alarm_create(self):
        self.fc.alarm.create.assert_called_once_with({'alarm_actions': [], 'description': 'Do stuff with gnocchi aggregation by resource', 'enabled': True, 'insufficient_data_actions': [], 'ok_actions': [], 'name': mock.ANY, 'type': 'gnocchi_aggregation_by_resources_threshold', 'repeat_actions': True, 'gnocchi_aggregation_by_resources_threshold_rule': {'aggregation_method': 'mean', 'granularity': 60, 'evaluation_periods': 1, 'threshold': 50, 'comparison_operator': 'gt', 'metric': 'cpu_util', 'resource_type': 'instance', 'query': '{"=": {"server_group": "my_autoscaling_group"}}'}, 'time_constraints': [], 'severity': 'low'})

    def create_alarm(self):
        self.patchobject(aodh.AodhClientPlugin, '_create').return_value = self.fc
        self.fc.alarm.create.return_value = FakeAodhAlarm
        self.tmpl = template_format.parse(gnocchi_aggregation_by_resources_alarm_template)
        self.stack = utils.parse_stack(self.tmpl)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        return gnocchi.AodhGnocchiAggregationByResourcesAlarm('GnoAggregationByResourcesAlarm', resource_defns['GnoAggregationByResourcesAlarm'], self.stack)

    def test_update(self):
        rsrc = self.create_alarm()
        scheduler.TaskRunner(rsrc.create)()
        self._check_alarm_create()
        snippet = self.tmpl['resources']['GnoAggregationByResourcesAlarm']
        props = snippet['properties'].copy()
        props['query'] = '{"=": {"server_group": "my_new_group"}}'
        update_template = rsrc.t.freeze(properties=props)
        scheduler.TaskRunner(rsrc.update, update_template)()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
        self.fc.alarm.update.assert_called_once_with('foo', {'alarm_actions': [], 'description': 'Do stuff with gnocchi aggregation by resource', 'enabled': True, 'insufficient_data_actions': [], 'ok_actions': [], 'repeat_actions': True, 'gnocchi_aggregation_by_resources_threshold_rule': {'aggregation_method': 'mean', 'granularity': 60, 'evaluation_periods': 1, 'threshold': 50, 'comparison_operator': 'gt', 'metric': 'cpu_util', 'resource_type': 'instance', 'query': '{"=": {"server_group": "my_new_group"}}'}, 'time_constraints': [], 'severity': 'low'})

    def _prepare_resource(self, for_check=True):
        snippet = template_format.parse(gnocchi_aggregation_by_resources_alarm_template)
        self.stack = utils.parse_stack(snippet)
        res = self.stack['GnoAggregationByResourcesAlarm']
        if for_check:
            res.state_set(res.CREATE, res.COMPLETE)
        res.client = mock.Mock()
        mock_alarm = mock.Mock(enabled=True, state='ok')
        res.client().alarm.get.return_value = mock_alarm
        return res

    def test_show_resource(self):
        res = self._prepare_resource(for_check=False)
        res.client().alarm.create.return_value = FakeAodhAlarm
        res.client().alarm.get.return_value = FakeAodhAlarm
        scheduler.TaskRunner(res.create)()
        self.assertEqual(FakeAodhAlarm, res.FnGetAtt('show'))

    def test_gnocchi_alarm_aggr_by_resources_live_state(self):
        snippet = template_format.parse(gnocchi_aggregation_by_resources_alarm_template)
        self.stack = utils.parse_stack(snippet)
        resource_defns = self.stack.t.resource_definitions(self.stack)
        self.rsrc_defn = resource_defns['GnoAggregationByResourcesAlarm']
        self.client = mock.Mock()
        self.patchobject(gnocchi.AodhGnocchiAggregationByResourcesAlarm, 'client', return_value=self.client)
        alarm_res = gnocchi.AodhGnocchiAggregationByResourcesAlarm('alarm', self.rsrc_defn, self.stack)
        alarm_res.create()
        value = {'description': 'Do stuff with gnocchi aggregation by resource', 'alarm_actions': [], 'time_constraints': [], 'gnocchi_aggregation_by_resources_threshold_rule': {'metric': 'cpu_util', 'resource_type': 'instance', 'query': "{'=': {'server_group': 'my_autoscaling_group'}}", 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt'}}
        self.client.alarm.get.return_value = value
        expected_data = {'description': 'Do stuff with gnocchi aggregation by resource', 'alarm_actions': [], 'metric': 'cpu_util', 'resource_type': 'instance', 'query': "{'=': {'server_group': 'my_autoscaling_group'}}", 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt', 'insufficient_data_actions': None, 'enabled': None, 'ok_actions': None, 'repeat_actions': None, 'severity': None}
        reality = alarm_res.get_live_state(alarm_res.properties)
        self.assertEqual(expected_data, reality)