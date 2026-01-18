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
def test_rule_engine_not_any_of_regex_verify_fail(self):
    """Should deny authorization.

        The email in the assertion will fail the regex test.
        It is set to reject any @example.org address, but the
        incoming value is set to evildeveloper@example.org.
        RuleProcessor should yield ValidationError.

        """
    mapping = mapping_fixtures.MAPPING_DEVELOPER_REGEX
    assertion = mapping_fixtures.BAD_DEVELOPER_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    self.assertRaises(exception.ValidationError, rp.process, assertion)