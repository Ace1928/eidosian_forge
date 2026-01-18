from unittest import mock
from oslo_config import cfg
import uuid
from heat.db import api as db_api
from heat.db import models
from heat.engine import event
from heat.engine import stack
from heat.engine import template
from heat.objects import event as event_object
from heat.objects import resource_properties_data as rpd_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
def test_store_caps_events_random_purge(self):
    cfg.CONF.set_override('event_purge_batch_size', 100)
    cfg.CONF.set_override('max_events_per_stack', 1)
    self.resource.resource_id_set('resource_physical_id')
    e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'arkansas', None, None, self.resource.name, self.resource.type())
    e.store()
    with mock.patch('random.uniform') as mock_random_uniform:
        mock_random_uniform.return_value = 2.0 / 100 - 0.0001
        e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'alaska', None, None, self.resource.name, self.resource.type())
        e.store()
    events = event_object.Event.get_all_by_stack(self.ctx, self.stack.id)
    self.assertEqual(1, len(events))
    self.assertEqual('alaska', events[0].physical_resource_id)
    with mock.patch('random.uniform') as mock_random_uniform:
        mock_random_uniform.return_value = 2.0 / 100 + 0.0001
        e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'aardvark', None, None, self.resource.name, self.resource.type())
        e.store()
    events = event_object.Event.get_all_by_stack(self.ctx, self.stack.id)
    self.assertEqual(2, len(events))