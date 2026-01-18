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
def test_opt_out_authenticate_event(self):
    """Test that authenticate events are successfully opted out."""
    resource_type = EXP_RESOURCE_TYPE
    action = CREATED_OPERATION + '.' + resource_type
    initiator = mock
    target = mock
    outcome = 'success'
    event_type = 'identity.authenticate'
    meter_name = '%s.%s' % (event_type, outcome)
    conf = self.useFixture(config_fixture.Config(CONF))
    conf.config(notification_opt_out=[meter_name])
    with mock.patch.object(notifications._get_notifier(), 'info') as mocked:
        notifications._send_audit_notification(action, initiator, outcome, target, event_type)
        mocked.assert_not_called()