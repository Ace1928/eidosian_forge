import copy
import uuid
from keystone.exception import ValidationError
from keystone.federation import utils
from keystone.tests import unit
def test_rule_processor_extract_projects_schema1_0(self):
    projects_list = [{'name': 'project1', 'domain': self.domain_mock}]
    identity_values = {'projects': projects_list}
    result = self.rule_processor.extract_projects(identity_values)
    self.assertEqual(projects_list, result)