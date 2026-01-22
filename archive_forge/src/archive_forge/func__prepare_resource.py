from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.resources.openstack.aodh.gnocchi import alarm as gnocchi
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
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