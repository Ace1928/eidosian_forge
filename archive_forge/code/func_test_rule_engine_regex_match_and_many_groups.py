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
def test_rule_engine_regex_match_and_many_groups(self):
    """Should return group DEVELOPER_GROUP_ID and TESTER_GROUP_ID.

        The TESTER_ASSERTION should successfully have a match in
        MAPPING_LARGE. This will test a successful regex match
        for an `any_one_of` evaluation type, and will have many
        groups returned.

        """
    self._rule_engine_regex_match_and_many_groups(mapping_fixtures.TESTER_ASSERTION)