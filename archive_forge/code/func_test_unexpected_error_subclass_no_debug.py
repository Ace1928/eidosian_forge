import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_unexpected_error_subclass_no_debug(self):
    self.config_fixture.config(debug=False)
    e = UnexpectedExceptionTestCase.SubClassExc(debug_info=self.exc_str)
    self.assertEqual(exception.UnexpectedError.message_format, str(e))