import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_unexpected_error_custom_message_binary_debug(self):
    self.config_fixture.config(debug=True, insecure_debug=True)
    binary_msg = b'something'
    e = exception.UnexpectedError(binary_msg)
    self.assertEqual('%s %s' % (str(binary_msg), exception.SecurityError.amendment), str(e))