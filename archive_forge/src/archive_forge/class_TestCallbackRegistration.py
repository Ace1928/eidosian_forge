import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
class TestCallbackRegistration(unit.BaseTestCase):

    def setUp(self):
        super(TestCallbackRegistration, self).setUp()
        self.mock_log = mock.Mock()
        self.mock_log.logger.getEffectiveLevel.return_value = log.DEBUG

    def verify_log_message(self, data):
        """Verify log message.

        Tests that use this are a little brittle because adding more
        logging can break them.

        TODO(dstanek): remove the need for this in a future refactoring

        """
        log_fn = self.mock_log.debug
        self.assertEqual(len(data), log_fn.call_count)
        for datum in data:
            log_fn.assert_any_call(mock.ANY, datum)

    def test_a_function_callback(self):

        def callback(*args, **kwargs):
            pass
        resource_type = 'thing'
        with mock.patch('keystone.notifications.LOG', self.mock_log):
            notifications.register_event_callback(CREATED_OPERATION, resource_type, callback)
        callback = 'keystone.tests.unit.common.test_notifications.callback'
        expected_log_data = {'callback': callback, 'event': 'identity.%s.created' % resource_type}
        self.verify_log_message([expected_log_data])

    def test_a_method_callback(self):

        class C(object):

            def callback(self, *args, **kwargs):
                pass
        with mock.patch('keystone.notifications.LOG', self.mock_log):
            notifications.register_event_callback(CREATED_OPERATION, 'thing', C().callback)
        callback = 'keystone.tests.unit.common.test_notifications.C.callback'
        expected_log_data = {'callback': callback, 'event': 'identity.thing.created'}
        self.verify_log_message([expected_log_data])

    def test_a_list_of_callbacks(self):

        def callback(*args, **kwargs):
            pass

        class C(object):

            def callback(self, *args, **kwargs):
                pass
        with mock.patch('keystone.notifications.LOG', self.mock_log):
            notifications.register_event_callback(CREATED_OPERATION, 'thing', [callback, C().callback])
        callback_1 = 'keystone.tests.unit.common.test_notifications.callback'
        callback_2 = 'keystone.tests.unit.common.test_notifications.C.callback'
        expected_log_data = [{'callback': callback_1, 'event': 'identity.thing.created'}, {'callback': callback_2, 'event': 'identity.thing.created'}]
        self.verify_log_message(expected_log_data)

    def test_an_invalid_callback(self):
        self.assertRaises(TypeError, notifications.register_event_callback, (CREATED_OPERATION, 'thing', object()))

    def test_an_invalid_event(self):

        def callback(*args, **kwargs):
            pass
        self.assertRaises(ValueError, notifications.register_event_callback, uuid.uuid4().hex, 'thing', callback)