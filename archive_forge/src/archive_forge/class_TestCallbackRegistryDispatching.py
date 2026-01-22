from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
class TestCallbackRegistryDispatching(base.BaseTestCase):

    def setUp(self):
        super(TestCallbackRegistryDispatching, self).setUp()
        self.callback_manager = mock.Mock()
        self.registry_fixture = fixture.CallbackRegistryFixture(callback_manager=self.callback_manager)
        self.useFixture(self.registry_fixture)

    def test_subscribe(self):
        registry.subscribe(my_callback, 'my-resource', 'my-event')
        self.callback_manager.subscribe.assert_called_with(my_callback, 'my-resource', 'my-event', priority_group.PRIORITY_DEFAULT, False)

    def test_subscribe_explicit_priority(self):
        registry.subscribe(my_callback, 'my-resource', 'my-event', PRI_CALLBACK)
        self.callback_manager.subscribe.assert_called_with(my_callback, 'my-resource', 'my-event', PRI_CALLBACK, False)

    def test_unsubscribe(self):
        registry.unsubscribe(my_callback, 'my-resource', 'my-event')
        self.callback_manager.unsubscribe.assert_called_with(my_callback, 'my-resource', 'my-event')

    def test_unsubscribe_by_resource(self):
        registry.unsubscribe_by_resource(my_callback, 'my-resource')
        self.callback_manager.unsubscribe_by_resource.assert_called_with(my_callback, 'my-resource')

    def test_unsubscribe_all(self):
        registry.unsubscribe_all(my_callback)
        self.callback_manager.unsubscribe_all.assert_called_with(my_callback)

    def test_clear(self):
        registry.clear()
        self.callback_manager.clear.assert_called_with()

    def test_publish_payload(self):
        event_payload = events.EventPayload(mock.ANY)
        registry.publish('x', 'y', self, payload=event_payload)
        self.callback_manager.publish.assert_called_with('x', 'y', self, payload=event_payload)