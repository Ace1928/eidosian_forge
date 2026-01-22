import flask
import uuid
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.auth.plugins import mapped
import keystone.conf
from keystone import exception
from keystone.federation import utils as mapping_utils
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from unittest import mock
class TestUnicodeAssertionData(unit.BaseTestCase):
    """Ensure that unicode data in the assertion headers works.

    Bug #1525250 reported that something was not getting correctly encoded
    and/or decoded when assertion data contained non-ASCII characters.

    This test class mimics what happens in a real HTTP request.
    """

    def setUp(self):
        super(TestUnicodeAssertionData, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(group='federation', assertion_prefix='PFX')

    def _pull_mapping_rules_from_the_database(self):
        return jsonutils.loads(jsonutils.dumps(mapping_fixtures.MAPPING_UNICODE))

    def _pull_assertion_from_the_request_headers(self):
        app = flask.Flask(__name__)
        with app.test_request_context(path='/path', environ_overrides=mapping_fixtures.UNICODE_NAME_ASSERTION):
            data = mapping_utils.get_assertion_params_from_env()
            return dict(data)

    def test_unicode(self):
        mapping = self._pull_mapping_rules_from_the_database()
        assertion = self._pull_assertion_from_the_request_headers()
        rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
        values = rp.process(assertion)
        fn = assertion.get('PFX_FirstName')
        ln = assertion.get('PFX_LastName')
        full_name = '%s %s' % (fn, ln)
        user_name = values.get('user', {}).get('name')
        self.assertEqual(full_name, user_name)