from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as mistral_client_plugin
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from oslo_serialization import jsonutils
class ZaqarSubscriptionTest(common.HeatTestCase):

    def setUp(self):
        super(ZaqarSubscriptionTest, self).setUp()
        self.fc = mock.Mock()
        self.patchobject(resource.Resource, 'client', return_value=self.fc)
        self.ctx = utils.dummy_context()
        self.subscr_id = '58138648c1e2eb7355d62137'

    def parse_stack(self, t):
        stack_name = 'test_stack'
        tmpl = template.Template(t)
        self.stack = stack.Stack(self.ctx, stack_name, tmpl)
        self.stack.validate()
        self.stack.store()

    def test_validate_subscriber_type(self):
        t = template_format.parse(subscr_template)
        t['Resources']['MySubscription']['Properties']['subscriber'] = 'foo:ba'
        stack_name = 'test_stack'
        tmpl = template.Template(t)
        self.stack = stack.Stack(self.ctx, stack_name, tmpl)
        exc = self.assertRaises(exception.StackValidationFailed, self.stack.validate)
        self.assertEqual('The subscriber type of must be one of: http, https, mailto, trust+http, trust+https.', str(exc))

    def test_create(self):
        t = template_format.parse(subscr_template)
        self.parse_stack(t)
        subscr = self.stack['MySubscription']
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.return_value = fake_subscr
        scheduler.TaskRunner(subscr.create)()
        self.assertEqual(self.subscr_id, subscr.FnGetRefId())
        self.fc.subscription.assert_called_once_with(subscr.properties['queue_name'], options={'key1': 'value1'}, subscriber=u'mailto:name@domain.com', ttl=3600)

    def test_delete(self):
        t = template_format.parse(subscr_template)
        self.parse_stack(t)
        subscr = self.stack['MySubscription']
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.return_value = fake_subscr
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options={'key1': 'value1'}, subscriber=u'mailto:name@domain.com', ttl=3600)
        scheduler.TaskRunner(subscr.delete)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], id=self.subscr_id, auto_create=False)
        self.assertTrue(getattr(fake_subscr, '_deleted', False))

    def test_delete_not_found(self):
        t = template_format.parse(subscr_template)
        self.parse_stack(t)
        subscr = self.stack['MySubscription']
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.side_effect = [fake_subscr, ResourceNotFound]
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options={'key1': 'value1'}, subscriber=u'mailto:name@domain.com', ttl=3600)
        scheduler.TaskRunner(subscr.delete)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], id=self.subscr_id, auto_create=False)

    def test_update_in_place(self):
        t = template_format.parse(subscr_template)
        self.parse_stack(t)
        subscr = self.stack['MySubscription']
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.return_value = fake_subscr
        t = template_format.parse(subscr_template)
        new_subscr = t['Resources']['MySubscription']
        new_subscr['Properties']['ttl'] = '3601'
        resource_defns = template.Template(t).resource_definitions(self.stack)
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options={'key1': 'value1'}, subscriber=u'mailto:name@domain.com', ttl=3600)
        scheduler.TaskRunner(subscr.update, resource_defns['MySubscription'])()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], id=self.subscr_id, auto_create=False)

    def test_update_replace(self):
        t = template_format.parse(subscr_template)
        self.parse_stack(t)
        subscr = self.stack['MySubscription']
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        self.fc.subscription.return_value = fake_subscr
        t = template_format.parse(subscr_template)
        t['Resources']['MySubscription']['Properties']['queue_name'] = 'foo'
        resource_defns = template.Template(t).resource_definitions(self.stack)
        new_subscr = resource_defns['MySubscription']
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options={'key1': 'value1'}, subscriber=u'mailto:name@domain.com', ttl=3600)
        err = self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(subscr.update, new_subscr))
        msg = 'The Resource MySubscription requires replacement.'
        self.assertEqual(msg, str(err))

    def test_show_resource(self):
        t = template_format.parse(subscr_template)
        self.parse_stack(t)
        subscr = self.stack['MySubscription']
        fake_subscr = FakeSubscription(subscr.properties['queue_name'], self.subscr_id)
        props = t['Resources']['MySubscription']['Properties']
        fake_subscr.ttl = props['ttl']
        fake_subscr.subscriber = props['subscriber']
        fake_subscr.options = props['options']
        self.fc.subscription.return_value = fake_subscr
        rsrc_data = props.copy()
        rsrc_data['id'] = self.subscr_id
        scheduler.TaskRunner(subscr.create)()
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], options={'key1': 'value1'}, subscriber=u'mailto:name@domain.com', ttl=3600)
        self.assertEqual(rsrc_data, subscr._show_resource())
        self.assertEqual({'queue_name': props['queue_name'], 'subscriber': props['subscriber'], 'ttl': props['ttl'], 'options': props['options']}, subscr.parse_live_resource_data(subscr.properties, subscr._show_resource()))
        self.fc.subscription.assert_called_with(subscr.properties['queue_name'], id=self.subscr_id, auto_create=False)