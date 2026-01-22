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
class EventEncryptedTest(EventCommon):

    def setUp(self):
        super(EventEncryptedTest, self).setUp()
        self._setup_stack(tmpl, encrypted=True)

    def test_props_encrypted(self):
        e = event.Event(self.ctx, self.stack, 'TEST', 'IN_PROGRESS', 'Testing', 'wibble', self.resource._rsrc_prop_data_id, self.resource._stored_properties_data, self.resource.name, self.resource.type())
        e.store()
        e_obj = event_object.Event.get_all_by_stack(self.resource.context, self.stack.id)[0]
        rpd_id = e_obj['rsrc_prop_data_id']
        with db_api.context_manager.reader.using(self.resource.context):
            results = self.resource.context.session.query(models.ResourcePropertiesData).filter_by(id=rpd_id)
        self.assertNotEqual('goo', results[0]['data']['Foo'])
        self.assertTrue(results[0]['encrypted'])
        ev = event_object.Event.get_all_by_stack(self.ctx, self.stack.id)[0]
        self.assertIsNone(ev._resource_properties)
        self.assertEqual({'Foo': 'goo'}, ev.resource_properties)
        filters = {'uuid': ev.uuid}
        ev = event_object.Event.get_all_by_stack(self.ctx, self.stack.id, filters=filters)[0]
        self.assertIsNotNone(ev._resource_properties)
        self.assertEqual({'Foo': 'goo'}, ev.resource_properties)