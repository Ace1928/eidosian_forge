import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_normalize_user_unexpected_type_schema_2_0(self):
    user = {'type': 'weird-type'}
    self.assertRaises(ValidationError, self.rule_processor_schema_2_0.normalize_user, user, self.domain_mock)