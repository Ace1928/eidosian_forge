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
def test_store_caps_resource_props_data(self):
    cfg.CONF.set_override('event_purge_batch_size', 2)
    cfg.CONF.set_override('max_events_per_stack', 3)
    self.resource.resource_id_set('resource_physical_id')
    e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'alabama', self.resource._rsrc_prop_data_id, self.resource._stored_properties_data, self.resource.name, self.resource.type())
    e.store()
    rpd1_id = self.resource._rsrc_prop_data_id
    rpd2 = rpd_object.ResourcePropertiesData.create(self.ctx, {'encrypted': False, 'data': {'foo': 'bar'}})
    rpd2_id = rpd2.id
    e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'arizona', rpd2_id, rpd2.data, self.resource.name, self.resource.type())
    e.store()
    rpd3 = rpd_object.ResourcePropertiesData.create(self.ctx, {'encrypted': False, 'data': {'foo': 'bar'}})
    rpd3_id = rpd3.id
    e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'arkansas', rpd3_id, rpd3.data, self.resource.name, self.resource.type())
    e.store()
    rpd4 = rpd_object.ResourcePropertiesData.create(self.ctx, {'encrypted': False, 'data': {'foo': 'bar'}})
    rpd4_id = rpd4.id
    e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'arkansas', rpd4_id, rpd4.data, self.resource.name, self.resource.type())
    e.store()
    events = event_object.Event.get_all_by_stack(self.ctx, self.stack.id)
    self.assertEqual(2, len(events))
    self.assertEqual('arkansas', events[0].physical_resource_id)
    with db_api.context_manager.reader.using(self.ctx):
        self.assertIsNotNone(self.ctx.session.get(models.ResourcePropertiesData, rpd1_id))
        self.assertIsNone(self.ctx.session.get(models.ResourcePropertiesData, rpd2_id))
        self.assertIsNotNone(self.ctx.session.get(models.ResourcePropertiesData, rpd3_id))
        self.assertIsNotNone(self.ctx.session.get(models.ResourcePropertiesData, rpd4_id))