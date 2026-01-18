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
def test_rule_engine_discards_nonstring_objects(self):
    """Check whether RuleProcessor discards non string objects.

        Despite the fact that assertion is malformed and contains
        non string objects, RuleProcessor should correctly discard them and
        successfully have a match in MAPPING_LARGE.

        """
    self._rule_engine_regex_match_and_many_groups(mapping_fixtures.MALFORMED_TESTER_ASSERTION)