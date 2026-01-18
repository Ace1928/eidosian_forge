import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_unicode_string(self):
    e = exception.ValidationError(attribute='xx', target='Long â\x80\x93 Dash')
    self.assertIn('Long â\x80\x93 Dash', str(e))