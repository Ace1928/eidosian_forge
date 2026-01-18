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
def test_rule_engine_no_regex_match(self):
    """Should deny authorization, the email of the tester won't match.

        This will not match since the email in the assertion will fail
        the regex test. It is set to match any @example.com address.
        But the incoming value is set to eviltester@example.org.
        RuleProcessor should raise ValidationError.

        """
    mapping = mapping_fixtures.MAPPING_LARGE
    assertion = mapping_fixtures.BAD_TESTER_ASSERTION
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    self.assertRaises(exception.ValidationError, rp.process, assertion)