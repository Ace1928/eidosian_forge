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
def test_rule_engine_fails_after_discarding_nonstring(self):
    """Check whether RuleProcessor discards non string objects.

        Expect RuleProcessor to discard non string object, which
        is required for a correct rule match. RuleProcessor will result with
        ValidationError.

        """
    mapping = mapping_fixtures.MAPPING_SMALL
    rp = mapping_utils.RuleProcessor(FAKE_MAPPING_ID, mapping['rules'])
    assertion = mapping_fixtures.CONTRACTOR_MALFORMED_ASSERTION
    self.assertRaises(exception.ValidationError, rp.process, assertion)