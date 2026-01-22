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
class AuditNotificationsTestCase(unit.BaseTestCase):

    def setUp(self):
        super(AuditNotificationsTestCase, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.addCleanup(notifications.clear_subscribers)

    def _test_notification_operation_with_basic_format(self, notify_function, operation):
        self.config_fixture.config(notification_format='basic')
        exp_resource_id = uuid.uuid4().hex
        callback = register_callback(operation)
        notify_function(EXP_RESOURCE_TYPE, exp_resource_id)
        callback.assert_called_once_with('identity', EXP_RESOURCE_TYPE, operation, {'resource_info': exp_resource_id})

    def _test_notification_operation_with_cadf_format(self, notify_function, operation):
        self.config_fixture.config(notification_format='cadf')
        exp_resource_id = uuid.uuid4().hex
        with mock.patch('keystone.notifications._create_cadf_payload') as cadf_notify:
            notify_function(EXP_RESOURCE_TYPE, exp_resource_id)
            initiator = None
            reason = None
            cadf_notify.assert_called_once_with(operation, EXP_RESOURCE_TYPE, exp_resource_id, notifications.taxonomy.OUTCOME_SUCCESS, initiator, reason)
            notify_function(EXP_RESOURCE_TYPE, exp_resource_id, public=False)
            cadf_notify.assert_called_once_with(operation, EXP_RESOURCE_TYPE, exp_resource_id, notifications.taxonomy.OUTCOME_SUCCESS, initiator, reason)

    def test_resource_created_notification(self):
        self._test_notification_operation_with_basic_format(notifications.Audit.created, CREATED_OPERATION)
        self._test_notification_operation_with_cadf_format(notifications.Audit.created, CREATED_OPERATION)

    def test_resource_updated_notification(self):
        self._test_notification_operation_with_basic_format(notifications.Audit.updated, UPDATED_OPERATION)
        self._test_notification_operation_with_cadf_format(notifications.Audit.updated, UPDATED_OPERATION)

    def test_resource_deleted_notification(self):
        self._test_notification_operation_with_basic_format(notifications.Audit.deleted, DELETED_OPERATION)
        self._test_notification_operation_with_cadf_format(notifications.Audit.deleted, DELETED_OPERATION)

    def test_resource_disabled_notification(self):
        self._test_notification_operation_with_basic_format(notifications.Audit.disabled, DISABLED_OPERATION)
        self._test_notification_operation_with_cadf_format(notifications.Audit.disabled, DISABLED_OPERATION)