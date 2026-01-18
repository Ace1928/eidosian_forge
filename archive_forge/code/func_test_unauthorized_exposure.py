import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_unauthorized_exposure(self):
    self.config_fixture.config(debug=False)
    risky_info = uuid.uuid4().hex
    e = exception.Unauthorized(message=risky_info)
    self.assertValidJsonRendering(e)
    self.assertNotIn(risky_info, str(e))