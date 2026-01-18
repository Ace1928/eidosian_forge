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
def test_password_expired_sends_notification(self):
    password = uuid.uuid4().hex
    password_creation_time = datetime.datetime.utcnow() - datetime.timedelta(days=CONF.security_compliance.password_expires_days + 1)
    freezer = freezegun.freeze_time(password_creation_time)
    freezer.start()
    user_ref = unit.new_user_ref(domain_id=self.domain_id, password=password)
    user_ref = PROVIDERS.identity_api.create_user(user_ref)
    with self.make_request():
        PROVIDERS.identity_api.authenticate(user_ref['id'], password)
    freezer.stop()
    reason_type = exception.PasswordExpired.message_format % {'user_id': user_ref['id']}
    expected_reason = {'reasonCode': '401', 'reasonType': reason_type}
    with self.make_request():
        self.assertRaises(exception.PasswordExpired, PROVIDERS.identity_api.authenticate, user_id=user_ref['id'], password=password)
    self._assert_last_audit(None, 'authenticate', None, cadftaxonomy.ACCOUNT_USER, reason=expected_reason)