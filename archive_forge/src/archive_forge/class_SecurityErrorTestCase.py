import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
class SecurityErrorTestCase(ExceptionTestCase):
    """Test whether security-related info is exposed to the API user."""

    def setUp(self):
        super(SecurityErrorTestCase, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(public_endpoint='http://localhost:5050')

    def test_unauthorized_exposure(self):
        self.config_fixture.config(debug=False)
        risky_info = uuid.uuid4().hex
        e = exception.Unauthorized(message=risky_info)
        self.assertValidJsonRendering(e)
        self.assertNotIn(risky_info, str(e))

    def test_unauthorized_exposure_in_debug(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        risky_info = uuid.uuid4().hex
        e = exception.Unauthorized(message=risky_info)
        self.assertValidJsonRendering(e)
        self.assertIn(risky_info, str(e))

    def test_forbidden_exposure(self):
        self.config_fixture.config(debug=False)
        risky_info = uuid.uuid4().hex
        e = exception.Forbidden(message=risky_info)
        self.assertValidJsonRendering(e)
        self.assertNotIn(risky_info, str(e))

    def test_forbidden_exposure_in_debug(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        risky_info = uuid.uuid4().hex
        e = exception.Forbidden(message=risky_info)
        self.assertValidJsonRendering(e)
        self.assertIn(risky_info, str(e))

    def test_forbidden_action_exposure(self):
        self.config_fixture.config(debug=False)
        risky_info = uuid.uuid4().hex
        action = uuid.uuid4().hex
        e = exception.ForbiddenAction(message=risky_info, action=action)
        self.assertValidJsonRendering(e)
        self.assertNotIn(risky_info, str(e))
        self.assertIn(action, str(e))
        self.assertNotIn(exception.SecurityError.amendment, str(e))
        e = exception.ForbiddenAction(action=action)
        self.assertValidJsonRendering(e)
        self.assertIn(action, str(e))
        self.assertNotIn(exception.SecurityError.amendment, str(e))

    def test_forbidden_action_exposure_in_debug(self):
        self.config_fixture.config(debug=True, insecure_debug=True)
        risky_info = uuid.uuid4().hex
        action = uuid.uuid4().hex
        e = exception.ForbiddenAction(message=risky_info, action=action)
        self.assertValidJsonRendering(e)
        self.assertIn(risky_info, str(e))
        self.assertIn(exception.SecurityError.amendment, str(e))
        e = exception.ForbiddenAction(action=action)
        self.assertValidJsonRendering(e)
        self.assertIn(action, str(e))
        self.assertNotIn(exception.SecurityError.amendment, str(e))

    def test_forbidden_action_no_message(self):
        action = uuid.uuid4().hex
        self.config_fixture.config(debug=False)
        e = exception.ForbiddenAction(action=action)
        exposed_message = str(e)
        self.assertIn(action, exposed_message)
        self.assertNotIn(exception.SecurityError.amendment, str(e))
        self.config_fixture.config(debug=True)
        e = exception.ForbiddenAction(action=action)
        self.assertEqual(exposed_message, str(e))

    def test_unicode_argument_message(self):
        self.config_fixture.config(debug=False)
        risky_info = u'继续行缩进或'
        e = exception.Forbidden(message=risky_info)
        self.assertValidJsonRendering(e)
        self.assertNotIn(risky_info, str(e))